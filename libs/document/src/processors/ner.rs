use async_trait::async_trait;
use orm::entity::{
    document,
    sea_orm_active_enums::{DocumentStatus, DocumentType},
};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter,
};

use crate::{
    document::{base::Document, text::Text},
    errors::DocumentErrors,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

/// Processor for text documents using the MuRIL Named Entity Recognition API.
///
/// Calls POST /api/ner with `{"text": "<string>"}` and writes the resulting entities
/// to `doc.update_extracted_information()`.
pub struct NerProcessor {
    base_url: String,
    client: reqwest::Client,
}

impl NerProcessor {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_default_url() -> Self {
        let base_url = std::env::var("POLICE_AI_BASE_URL")
            .unwrap_or_else(|_| "https://atharv1909--police-ai-engine-fastapi-app.modal.run".to_string());
        Self::new(base_url)
    }
}

#[async_trait]
impl DocumentProcessor for NerProcessor {
    async fn process(
        &self,
        doc: &dyn Document,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[NerProcessor][START] Processing doc_id={}", doc_id);

        // 1. Fetch raw bytes from storage / memory
        let bytes = doc.fetch().await.map_err(|e| {
            tracing::error!(
                "[NerProcessor][STORAGE_ERROR] Failed to fetch bytes for doc_id={}: {}",
                doc_id,
                e
            );
            e
        })?;

        // 2. Decode UTF-8 text
        let text = String::from_utf8_lossy(&bytes).to_string();
        if text.trim().is_empty() {
            tracing::warn!(
                "[NerProcessor] Empty text for doc_id={}, skipping NER extraction",
                doc_id
            );
            return Ok(());
        }

        // 3. Prepare payload according to schema: {"text": "<string>"}
        let payload = serde_json::json!({
            "text": text
        });

        let url = format!("{}/api/ner", self.base_url.trim_end_matches('/'));
        tracing::info!(
            "[NerProcessor][MODAL_API] Dispatching POST to {} for doc_id={}",
            url,
            doc_id
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(
                    "[NerProcessor][MODAL_API_ERROR] Network request failed for doc_id={}: {}",
                    doc_id,
                    e
                );
                DocumentErrors::StorageError(format!("Network request to /api/ner failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                "[NerProcessor][MODAL_API_ERROR] /api/ner returned non-success status={} for doc_id={}: {}",
                status,
                doc_id,
                body
            );
            return Err(DocumentErrors::StorageError(format!(
                "/api/ner returned HTTP {}: {}",
                status, body
            )));
        }

        // 4. Parse JSON response
        let mut extracted_json: serde_json::Value = response.json().await.map_err(|e| {
            tracing::error!(
                "[NerProcessor][JSON_PARSE_ERROR] Failed to parse JSON response for doc_id={}: {}",
                doc_id,
                e
            );
            DocumentErrors::StorageError(format!("Failed to parse /api/ner response: {}", e))
        })?;

        // Ensure transcribed_text is present in the object
        if let Some(map) = extracted_json.as_object_mut() {
            if !map.contains_key("transcribed_text") {
                map.insert("transcribed_text".to_string(), serde_json::Value::String(text));
            }
        }

        // 5. Update extracted information in DB
        doc.update_extracted_information(extracted_json, db)
            .await
            .map_err(|e| {
                tracing::error!(
                    "[NerProcessor][DB_ERROR] Failed to persist extracted_information for doc_id={}: {}",
                    doc_id,
                    e
                );
                e
            })?;

        tracing::info!(
            "[NerProcessor][FINISH] Successfully extracted and persisted NER entities for doc_id={}",
            doc_id
        );
        Ok(())
    }

    async fn cron_func(
        &self,
        db: &DatabaseConnection,
        store: &S3ObjectStore,
    ) -> Result<(), DocumentErrors> {
        tracing::debug!("[NerProcessor][CRON] Checking for unprocessed text documents...");

        let documents = document::Entity::find()
            .filter(document::Column::Type.eq(DocumentType::Text))
            .filter(document::Column::Status.eq(DocumentStatus::Success))
            .filter(document::Column::ExtractedInformation.is_null())
            .all(db)
            .await
            .map_err(|e| {
                tracing::error!(
                    "[NerProcessor][DB_ERROR] Failed to query unprocessed text documents: {}",
                    e
                );
                DocumentErrors::DatabaseError(e)
            })?;

        if documents.is_empty() {
            return Ok(());
        }

        tracing::info!(
            "[NerProcessor][CRON] Found {} unprocessed text document(s)",
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
                    "[NerProcessor][DB_ERROR] Failed to set status=Processing for doc_id={}: {}",
                    doc_id,
                    e
                );
                continue;
            }
            tracing::info!(
                "[NerProcessor][STATUS_TRANSITION] doc_id={} → Processing",
                doc_id
            );

            let text_doc = Text::new(doc_id, model.object_key.clone(), store.clone());

            // Execute process
            let process_res = self.process(&text_doc, db).await;
            if let Err(ref e) = process_res {
                tracing::error!(
                    "[NerProcessor][PROCESSING_ERROR] Error processing doc_id={}: {}",
                    doc_id,
                    e
                );
            }

            if let Ok(Some(doc)) = document::Entity::find_by_id(doc_id).one(db).await {
                let mut active: document::ActiveModel = doc.into();
                active.status = Set(if process_res.is_ok() {
                    DocumentStatus::Finish
                } else {
                    DocumentStatus::Failed
                });
                active.updated_at = Set(chrono::Utc::now().fixed_offset());
                if let Err(e) = active.update(db).await {
                    tracing::error!(
                        "[NerProcessor][DB_ERROR] Failed to set status for doc_id={}: {}",
                        doc_id,
                        e
                    );
                } else {
                    tracing::info!(
                        "[NerProcessor][STATUS_TRANSITION] doc_id={} -> Finish/Failed",
                        doc_id
                    );
                }
            }
        }

        Ok(())
    }
}
