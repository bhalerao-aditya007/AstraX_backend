use std::sync::Arc;

use async_trait::async_trait;
use orm::entity::{
    document,
    sea_orm_active_enums::{DocumentStatus, DocumentType},
};
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter,
};

use crate::{
    document::{base::Document, image::Image, transcribed::TranscribedDocument},
    errors::DocumentErrors,
    processors::{base::DocumentProcessor, ner::NerProcessor},
    storage::s3_object_store::S3ObjectStore,
};

/// End-to-End Pipeline processor for handwritten diary / FIR images using POST /api/pipeline.
///
/// Sends the image binary via multipart form data to POST /api/pipeline, retrieves the
/// transcribed text, and chains to `NerProcessor::process` to extract entities and persist
/// the result atomically to the database.
pub struct PipelineProcessor {
    base_url: String,
    client: reqwest::Client,
    ner_processor: Arc<NerProcessor>,
}

impl PipelineProcessor {
    pub fn new(base_url: String, ner_processor: Arc<NerProcessor>) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
            ner_processor,
        }
    }

    pub fn with_default_url(ner_processor: Arc<NerProcessor>) -> Self {
        let base_url = std::env::var("POLICE_AI_BASE_URL")
            .unwrap_or_else(|_| "https://atharv1909--police-ai-engine-fastapi-app.modal.run".to_string());
        Self::new(base_url, ner_processor)
    }
}

#[async_trait]
impl DocumentProcessor for PipelineProcessor {
    async fn process(
        &self,
        doc: &dyn Document,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[PipelineProcessor][START] Processing doc_id={}", doc_id);

        // 1. Fetch raw image bytes from storage
        let bytes = doc.fetch().await.map_err(|e| {
            tracing::error!(
                "[PipelineProcessor][STORAGE_ERROR] Failed to fetch image bytes for doc_id={}: {}",
                doc_id,
                e
            );
            e
        })?;

        let mime = doc.mime_type();

        // 2. Build multipart form data with field 'file'
        let file_part = reqwest::multipart::Part::bytes(bytes)
            .file_name("diary_page.jpg")
            .mime_str(mime)
            .map_err(|e| {
                tracing::error!(
                    "[PipelineProcessor][MIME_ERROR] Invalid mime type '{}' for doc_id={}: {}",
                    mime,
                    doc_id,
                    e
                );
                DocumentErrors::ValidationError(format!("Invalid mime type '{}': {}", mime, e))
            })?;

        let form = reqwest::multipart::Form::new().part("file", file_part);

        let url = format!("{}/api/pipeline", self.base_url.trim_end_matches('/'));
        tracing::info!(
            "[PipelineProcessor][MODAL_API] Dispatching multipart POST to {} for doc_id={}",
            url,
            doc_id
        );

        // 3. POST to /api/pipeline
        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(
                    "[PipelineProcessor][MODAL_API_ERROR] Network request failed for doc_id={}: {}",
                    doc_id,
                    e
                );
                DocumentErrors::StorageError(format!(
                    "Network request to /api/pipeline failed: {}",
                    e
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            tracing::error!(
                "[PipelineProcessor][MODAL_API_ERROR] /api/pipeline returned non-success status={} for doc_id={}: {}",
                status,
                doc_id,
                body
            );
            return Err(DocumentErrors::StorageError(format!(
                "/api/pipeline returned HTTP {}: {}",
                status, body
            )));
        }

        // 4. Parse pipeline response
        let pipeline_json: serde_json::Value = response.json().await.map_err(|e| {
            tracing::error!(
                "[PipelineProcessor][JSON_PARSE_ERROR] Failed to parse JSON response for doc_id={}: {}",
                doc_id,
                e
            );
            DocumentErrors::StorageError(format!("Failed to parse /api/pipeline response: {}", e))
        })?;

        let transcribed_text = pipeline_json
            .get("transcribed_text")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string();

        tracing::info!(
            "[PipelineProcessor] Extracted {} chars of transcribed_text for doc_id={}",
            transcribed_text.len(),
            doc_id
        );

        // 5. Chain to NerProcessor using TranscribedDocument in-memory adapter
        let transcribed_doc = TranscribedDocument::new(doc_id, transcribed_text);
        tracing::info!(
            "[PipelineProcessor][CHAIN] Forwarding transcribed text to NerProcessor for doc_id={}",
            doc_id
        );

        self.ner_processor.process(&transcribed_doc, db).await?;

        tracing::info!(
            "[PipelineProcessor][FINISH] Completed Pipeline & NER extraction for doc_id={}",
            doc_id
        );
        Ok(())
    }

    async fn cron_func(
        &self,
        db: &DatabaseConnection,
        store: &S3ObjectStore,
    ) -> Result<(), DocumentErrors> {
        tracing::debug!("[PipelineProcessor][CRON] Checking for unprocessed image documents...");

        let documents = document::Entity::find()
            .filter(document::Column::Type.eq(DocumentType::Image))
            .filter(document::Column::Status.eq(DocumentStatus::Success))
            .filter(document::Column::ExtractedInformation.is_null())
            .all(db)
            .await
            .map_err(|e| {
                tracing::error!(
                    "[PipelineProcessor][DB_ERROR] Failed to query unprocessed image documents: {}",
                    e
                );
                DocumentErrors::DatabaseError(e)
            })?;

        if documents.is_empty() {
            return Ok(());
        }

        tracing::info!(
            "[PipelineProcessor][CRON] Found {} unprocessed image document(s)",
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
                    "[PipelineProcessor][DB_ERROR] Failed to set status=Processing for doc_id={}: {}",
                    doc_id,
                    e
                );
                continue;
            }
            tracing::info!(
                "[PipelineProcessor][STATUS_TRANSITION] doc_id={} → Processing",
                doc_id
            );

            let image_doc = Image::new(doc_id, model.object_key.clone(), store.clone());

            // Execute process
            let process_res = self.process(&image_doc, db).await;
            if let Err(ref e) = process_res {
                tracing::error!(
                    "[PipelineProcessor][PROCESSING_ERROR] Error processing doc_id={}: {}",
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
                        "[PipelineProcessor][DB_ERROR] Failed to set status for doc_id={}: {}",
                        doc_id,
                        e
                    );
                } else {
                    tracing::info!(
                        "[PipelineProcessor][STATUS_TRANSITION] doc_id={} -> Finish/Failed",
                        doc_id
                    );
                }
            }
        }

        Ok(())
    }
}
