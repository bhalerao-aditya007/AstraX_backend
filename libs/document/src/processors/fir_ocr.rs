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
    gradio::GradioClient,
    processors::{base::DocumentProcessor, ner::NerProcessor},
    storage::s3_object_store::S3ObjectStore,
};

pub struct FirOcrProcessor {
    client: Arc<GradioClient>,
    ner_processor: Arc<NerProcessor>,
}

impl FirOcrProcessor {
    pub fn new(client: Arc<GradioClient>, ner_processor: Arc<NerProcessor>) -> Self {
        Self { client, ner_processor }
    }
}

#[async_trait]
impl DocumentProcessor for FirOcrProcessor {
    async fn process(&self, doc: &dyn Document, db: &DatabaseConnection) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[FirOcrProcessor][START] Processing doc_id={}", doc_id);

        let bytes = doc.fetch().await?;
        let mime = doc.mime_type();

        let result = self.client.predict_with_file("fir_ocr", bytes, "fir.jpg", mime, vec![]).await?;
        let mut parsed = GradioClient::parse_result(result);

        let mut text_to_ner = String::new();
        if let Some(map) = parsed.as_object() {
            if let Some(t) = map.get("transcribed_text").and_then(|v| v.as_str()) {
                text_to_ner.push_str(t);
            }
            if let Some(n) = map.get("narrative").and_then(|v| v.as_str()) {
                if !text_to_ner.is_empty() {
                    text_to_ner.push('\n');
                }
                text_to_ner.push_str(n);
            }
        }

        if !text_to_ner.is_empty() {
            tracing::info!("[FirOcrProcessor][CHAIN] Forwarding text to NerProcessor for doc_id={}", doc_id);
            let transcribed_doc = TranscribedDocument::new(doc_id, text_to_ner);
            self.ner_processor.process(&transcribed_doc, db).await?;

            // Retrieve the NER results that were saved by the NerProcessor
            if let Some(model) = document::Entity::find_by_id(doc_id).one(db).await.map_err(DocumentErrors::DatabaseError)? {
                if let Some(ner_json) = model.extracted_information {
                    if let (serde_json::Value::Object(ref mut parsed_map), serde_json::Value::Object(ner_map)) = (&mut parsed, ner_json) {
                        for (k, v) in ner_map {
                            parsed_map.insert(k, v);
                        }
                    }
                }
            }
        }

        doc.update_extracted_information(parsed, db).await?;
        tracing::info!("[FirOcrProcessor][FINISH] Completed FIR processing for doc_id={}", doc_id);
        Ok(())
    }

    async fn cron_func(&self, db: &DatabaseConnection, store: &S3ObjectStore) -> Result<(), DocumentErrors> {
        tracing::debug!("[FirOcrProcessor][CRON] Checking for unprocessed image documents...");
        let documents = document::Entity::find()
            .filter(document::Column::Type.eq(DocumentType::Image))
            .filter(document::Column::Status.eq(DocumentStatus::Success))
            .filter(document::Column::ExtractedInformation.is_null())
            .all(db)
            .await
            .map_err(|e| {
                tracing::error!("[FirOcrProcessor][DB_ERROR] Failed to query: {}", e);
                DocumentErrors::DatabaseError(e)
            })?;

        if documents.is_empty() {
            return Ok(());
        }
        tracing::info!("[FirOcrProcessor][CRON] Found {} unprocessed image document(s)", documents.len());

        for model in documents {
            let doc_id = model.id;
            let mut active: document::ActiveModel = model.clone().into();
            active.status = Set(DocumentStatus::Processing);
            active.updated_at = Set(chrono::Utc::now().fixed_offset());
            if let Err(e) = active.update(db).await {
                tracing::error!("[FirOcrProcessor][DB_ERROR] Failed to set status=Processing for doc_id={}: {}", doc_id, e);
                continue;
            }

            let image_doc = Image::new(doc_id, model.object_key.clone(), store.clone());
            if let Err(e) = self.process(&image_doc, db).await {
                tracing::error!("[FirOcrProcessor][PROCESSING_ERROR] Error for doc_id={}: {}", doc_id, e);
            }

            if let Ok(Some(d)) = document::Entity::find_by_id(doc_id).one(db).await {
                let mut active: document::ActiveModel = d.into();
                active.status = Set(DocumentStatus::Finish);
                active.updated_at = Set(chrono::Utc::now().fixed_offset());
                if let Err(e) = active.update(db).await {
                    tracing::error!("[FirOcrProcessor][DB_ERROR] Failed to set status=Finish for doc_id={}: {}", doc_id, e);
                } else {
                    tracing::info!("[FirOcrProcessor][STATUS_TRANSITION] doc_id={} -> Finish", doc_id);
                }
            }
        }
        Ok(())
    }
}
