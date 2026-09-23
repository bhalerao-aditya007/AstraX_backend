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
    document::{base::Document, text::Text},
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};
pub struct NerProcessor {
    client: Arc<GradioClient>,
}

impl NerProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }
    pub fn from_env() -> Self {
        Self::new(Arc::new(GradioClient::from_env()))
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

        // 3. Call the Gradio `ner` endpoint on the AstraX HF Space
        let payload = serde_json::json!({ "text": text });
        tracing::info!(
            "[NerProcessor][GRADIO] Dispatching 'ner' prediction for doc_id={}",
            doc_id
        );

        let result = self.client.predict_json("ner", payload).await.map_err(|e| {
            tracing::error!(
                "[NerProcessor][GRADIO_ERROR] 'ner' call failed for doc_id={}: {}",
                doc_id,
                e
            );
            e
        })?;

        let mut extracted_json = GradioClient::parse_result(result);

        // Ensure transcribed_text is present in the object
        if let Some(map) = extracted_json.as_object_mut() {
            if !map.contains_key("transcribed_text") {
                map.insert("transcribed_text".to_string(), serde_json::Value::String(text));
            }
        }

        // 4. Update extracted information in DB
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
            tracing::info!("[NerProcessor][STATUS_TRANSITION] doc_id={} → Processing", doc_id);

            let text_doc = Text::new(doc_id, model.object_key.clone(), store.clone());

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
