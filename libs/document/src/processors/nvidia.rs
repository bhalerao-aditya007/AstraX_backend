use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use orm::entity::{
    document,
    sea_orm_active_enums::{DocumentStatus, DocumentType},
};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter,
};
use serde::Deserialize;

use crate::{
    document::{base::Document, image::Image, text::Text, video::Video, voice::Voice},
    errors::DocumentErrors,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

// ---------------------------------------------------------------------------
// Specialized Forensic Prompts
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = "whatever you can extract in json give like that";

// ---------------------------------------------------------------------------
// Response Deserialization Types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

// ---------------------------------------------------------------------------
// NvidiaKimiProcessor
// ---------------------------------------------------------------------------

/// Universal Document Processor powered by NVIDIA NIM (Nemotron Omni / Kimi-K3).
///
/// Handles Image (OCR & visual forensics), Text (in-depth NER & conspiracy graphs),
/// and Voice (audio transcript analysis) documents.
pub struct NvidiaKimiProcessor {
    api_key: String,
    model: String,
    endpoint: String,
    client: reqwest::Client,
}

impl NvidiaKimiProcessor {
    /// Create a new processor with an explicit API key and model name.
    pub fn new(api_key: String, model: String) -> Self {
        let endpoint = std::env::var("NVIDIA_ENDPOINT")
            .unwrap_or_else(|_| "https://integrate.api.nvidia.com/v1/chat/completions".to_string());
        Self {
            api_key,
            model,
            endpoint,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(90))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Construct from environment variables:
    /// - `NVIDIA_API_KEY`: Required.
    /// - `NVIDIA_MODEL`: Defaults to `"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"`.
    /// - `NVIDIA_ENDPOINT`: Defaults to `"https://integrate.api.nvidia.com/v1/chat/completions"`.
    pub fn from_env() -> Result<Self, DocumentErrors> {
        let api_key = std::env::var("NVIDIA_API_KEY")
            .map_err(|_| DocumentErrors::StorageError("NVIDIA_API_KEY is not set".to_string()))?;
        let model = std::env::var("NVIDIA_MODEL")
            .unwrap_or_else(|_| "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning".to_string());
        Ok(Self::new(api_key, model))
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Clean response string and parse into `serde_json::Value`.
    /// Supports direct JSON objects, JSON arrays, and markdown-fenced code blocks.
    fn parse_model_json(content: &str) -> serde_json::Value {
        let trimmed = content.trim();

        // 1. Try direct parse
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return val;
        }

        // 2. Extract block within ```json ... ``` or ``` ... ```
        if let Some(start) = trimmed.find("```json") {
            let after_fence = &trimmed[start + 7..];
            if let Some(end) = after_fence.rfind("```") {
                let inner = after_fence[..end].trim();
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(inner) {
                    return val;
                }
            }
        } else if let Some(start) = trimmed.find("```") {
            let after_fence = &trimmed[start + 3..];
            if let Some(end) = after_fence.rfind("```") {
                let inner = after_fence[..end].trim();
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(inner) {
                    return val;
                }
            }
        }

        // 3. Extract substring between first '{' and last '}', or first '[' and last ']'
        let first_brace = trimmed.find('{');
        let last_brace = trimmed.rfind('}');
        let first_bracket = trimmed.find('[');
        let last_bracket = trimmed.rfind(']');

        let candidate = match (first_brace, last_brace, first_bracket, last_bracket) {
            (Some(fb), Some(lb), Some(fk), Some(lk)) => {
                if fb < fk && lb > lk {
                    Some(&trimmed[fb..=lb])
                } else {
                    Some(&trimmed[fk..=lk])
                }
            }
            (Some(fb), Some(lb), _, _) if fb <= lb => Some(&trimmed[fb..=lb]),
            (_, _, Some(fk), Some(lk)) if fk <= lk => Some(&trimmed[fk..=lk]),
            _ => None,
        };

        if let Some(json_str) = candidate {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(json_str) {
                return val;
            }
        }

        tracing::warn!(
            "[NvidiaProcessor] Model output was not valid JSON. Storing as raw_text."
        );
        serde_json::json!({
            "raw_text": content,
            "transcribed_text": content,
            "summary": "Extracted unstructured text from document."
        })
    }
}

pub type NvidiaProcessor = NvidiaKimiProcessor;

#[async_trait]
impl DocumentProcessor for NvidiaKimiProcessor {
    async fn process(
        &self,
        doc: &dyn Document,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[NvidiaProcessor][START] Processing doc_id={}", doc_id);

        // 1. Fetch document record from DB to determine its exact DocumentType
        let doc_model = document::Entity::find_by_id(doc_id)
            .one(db)
            .await
            .map_err(DocumentErrors::DatabaseError)?
            .ok_or_else(|| {
                DocumentErrors::NotFound(format!("Document with id={} not found", doc_id))
            })?;

        let doc_type = doc_model.r#type;
        tracing::info!(
            "[NvidiaProcessor] doc_id={} has type={:?}, mime='{}'",
            doc_id,
            doc_type,
            doc.mime_type()
        );

        // 2. Fetch raw file bytes from S3/MinIO
        let bytes = doc.fetch().await.map_err(|e| {
            tracing::error!(
                "[NvidiaProcessor][FETCH_ERROR] Failed to fetch bytes for doc_id={}: {}",
                doc_id,
                e
            );
            e
        })?;

        // 3. Build payload according to document type with requested system prompt
        let payload = match doc_type {
            DocumentType::Image => {
                let mime = doc.mime_type();
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                let data_url = format!("data:{};base64,{}", mime, b64);

                serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all information, text, entities, and details from this image into JSON."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 16384,
                    "reasoning_budget": 4096
                })
            }
            DocumentType::Text => {
                let text_content = String::from_utf8_lossy(&bytes);
                serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": format!("Extract all information, text, entities, and details from this document into JSON.\n\n{}", text_content)
                        }
                    ],
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 16384,
                    "reasoning_budget": 4096
                })
            }
            DocumentType::Voice => {
                let text_sample = String::from_utf8(bytes.clone());
                let prompt = match text_sample {
                    Ok(transcript) => format!(
                        "Extract all information, conversation transcript, entities, and details into JSON.\n\n{}",
                        transcript
                    ),
                    Err(_) => {
                        format!(
                            "Extract all information from this audio metadata into JSON.\nDocument Title: {}\nDescription: {}",
                            doc_model.title, doc_model.description
                        )
                    }
                };

                serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 16384,
                    "reasoning_budget": 4096
                })
            }
            DocumentType::Video => {
                let prompt = format!(
                    "Extract all surveillance details, timestamps, observed vehicles, persons, license plates, and incident chronology from this video evidence into JSON.\nDocument Title: {}\nDescription: {}",
                    doc_model.title, doc_model.description
                );

                serde_json::json!({
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "max_tokens": 16384,
                    "reasoning_budget": 4096
                })
            }
        };

        // 4. Dispatch request to NVIDIA NIM
        tracing::info!(
            "[NvidiaProcessor][HTTP_POST] Dispatching request to {} (model: '{}') for doc_id={}",
            self.endpoint,
            self.model,
            doc_id
        );

        let start_time = std::time::Instant::now();
        let mut response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(
                    "[NvidiaProcessor][NETWORK_ERROR] Request failed for doc_id={}: {}",
                    doc_id,
                    e
                );
                DocumentErrors::StorageError(format!("NVIDIA NIM request failed: {}", e))
            })?;

        // If 503 (worker local total limit reached), wait 3s and retry once, then fallback to moonshotai/kimi-k3 if still 503
        if response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            tracing::warn!(
                "[NvidiaProcessor][RETRY] 503 Service Unavailable received for doc_id={}. Retrying in 3s...",
                doc_id
            );
            tokio::time::sleep(Duration::from_secs(3)).await;
            let retry_resp = self
                .client
                .post(&self.endpoint)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .await;

            match retry_resp {
                Ok(resp) if resp.status().is_success() => {
                    response = resp;
                }
                _ => {
                    // Fallback to moonshotai/kimi-k3 if primary model worker pool is saturated
                    tracing::warn!(
                        "[NvidiaProcessor][FALLBACK] Primary model pool saturated. Falling back to moonshotai/kimi-k3 for doc_id={}...",
                        doc_id
                    );
                    let mut fallback_payload = payload.clone();
                    fallback_payload["model"] = serde_json::Value::String("moonshotai/kimi-k3".to_string());
                    if let Ok(fb_resp) = self
                        .client
                        .post(&self.endpoint)
                        .header("Authorization", format!("Bearer {}", self.api_key))
                        .header("Content-Type", "application/json")
                        .json(&fallback_payload)
                        .send()
                        .await
                    {
                        if fb_resp.status().is_success() {
                            response = fb_resp;
                        }
                    }
                }
            }
        }

        let elapsed = start_time.elapsed();

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                "[NvidiaProcessor][API_ERROR] NVIDIA API returned status={} for doc_id={} in {:?}: {}",
                status,
                doc_id,
                elapsed,
                body
            );
            return Err(DocumentErrors::StorageError(format!(
                "NVIDIA API returned HTTP {}: {}",
                status, body
            )));
        }

        tracing::info!(
            "[NvidiaProcessor][SUCCESS] NVIDIA API responded in {:?} for doc_id={}",
            elapsed,
            doc_id
        );

        // 5. Parse response content
        let completion: ChatCompletionResponse = response.json().await.map_err(|e| {
            tracing::error!(
                "[NvidiaProcessor][JSON_PARSE_ERROR] Failed to deserialize NVIDIA response for doc_id={}: {}",
                doc_id,
                e
            );
            DocumentErrors::StorageError(format!("Failed to parse NVIDIA response JSON: {}", e))
        })?;

        let content = completion
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .ok_or_else(|| {
                DocumentErrors::StorageError("No content returned in NVIDIA response choice".into())
            })?;

        let mut extracted = Self::parse_model_json(content);

        // Ensure transcribed_text field exists for downstream consumers if object
        if let Some(obj) = extracted.as_object_mut() {
            if !obj.contains_key("transcribed_text") {
                obj.insert(
                    "transcribed_text".to_string(),
                    serde_json::Value::String(content.to_string()),
                );
            }
        }

        // 6. Persist extracted JSON atomically into the database
        doc.update_extracted_information(extracted, db).await?;
        tracing::info!(
            "[NvidiaProcessor][FINISH] Successfully persisted extracted information for doc_id={}",
            doc_id
        );

        Ok(())
    }

    /// Background cron function (provided for full compliance with `DocumentProcessor`).
    /// Queries all unprocessed confirmed documents across Image, Text, and Voice types.
    async fn cron_func(
        &self,
        db: &DatabaseConnection,
        store: &S3ObjectStore,
    ) -> Result<(), DocumentErrors> {
        tracing::debug!("[NvidiaProcessor][CRON] Scanning for unprocessed documents...");

        // Query confirmed documents (or documents stuck in processing) where extracted_information IS NULL
        let documents = document::Entity::find()
            .filter(document::Column::ExtractedInformation.is_null())
            .filter(document::Column::Status.is_in([DocumentStatus::Success, DocumentStatus::Processing]))
            .all(db)
            .await
            .map_err(|e| {
                tracing::error!(
                    "[NvidiaProcessor][DB_ERROR] Failed to query unprocessed documents: {}",
                    e
                );
                DocumentErrors::DatabaseError(e)
            })?;

        if documents.is_empty() {
            return Ok(());
        }

        tracing::info!(
            "[NvidiaProcessor][CRON] Found {} unprocessed document(s)",
            documents.len()
        );

        for model in documents {
            let doc_id = model.id;

            // Transition status → Processing
            let mut active: document::ActiveModel = model.clone().into();
            active.status = Set(DocumentStatus::Processing);
            active.updated_at = Set(chrono::Utc::now().fixed_offset());
            if let Err(e) = active.update(db).await {
                tracing::error!(
                    "[NvidiaProcessor][DB_ERROR] Failed to set status=Processing for doc_id={}: {}",
                    doc_id,
                    e
                );
                continue;
            }

            // Wrap in matching Document implementation
            let process_result = match model.r#type {
                DocumentType::Image => {
                    let image_doc = Image::new(doc_id, model.object_key.clone(), store.clone());
                    self.process(&image_doc, db).await
                }
                DocumentType::Text => {
                    let text_doc = Text::new(doc_id, model.object_key.clone(), store.clone());
                    self.process(&text_doc, db).await
                }
                DocumentType::Voice => {
                    let voice_doc = Voice::new(doc_id, model.object_key.clone(), store.clone());
                    self.process(&voice_doc, db).await
                }
                DocumentType::Video => {
                    let video_doc = Video::new(doc_id, model.object_key.clone(), store.clone());
                    self.process(&video_doc, db).await
                }
            };

            match process_result {
                Ok(_) => {
                    // Transition status → Finish ONLY when processing succeeded and extracted_information is saved
                    if let Ok(Some(d)) = document::Entity::find_by_id(doc_id).one(db).await {
                        let mut active: document::ActiveModel = d.into();
                        active.status = Set(DocumentStatus::Finish);
                        active.updated_at = Set(chrono::Utc::now().fixed_offset());
                        let _ = active.update(db).await;
                        tracing::info!(
                            "[NvidiaProcessor][STATUS_TRANSITION] doc_id={} → Finish",
                            doc_id
                        );
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "[NvidiaProcessor][PROCESSING_ERROR] Error processing doc_id={}: {}",
                        doc_id,
                        e
                    );
                    // Reset status to Success so it will be retried on next cron tick instead of being stuck
                    if let Ok(Some(d)) = document::Entity::find_by_id(doc_id).one(db).await {
                        let mut active: document::ActiveModel = d.into();
                        active.status = Set(DocumentStatus::Success);
                        active.updated_at = Set(chrono::Utc::now().fixed_offset());
                        let _ = active.update(db).await;
                        tracing::warn!(
                            "[NvidiaProcessor][STATUS_RESET] doc_id={} reset to Success for retry",
                            doc_id
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
