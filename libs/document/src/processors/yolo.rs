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
    document::{base::Document, image::Image, video::Video},
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

pub struct YoloProcessor {
    client: Arc<GradioClient>,
}

impl YoloProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl DocumentProcessor for YoloProcessor {
    async fn process(&self, doc: &dyn Document, db: &DatabaseConnection) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[YoloProcessor][START] Processing doc_id={}", doc_id);

        let bytes = doc.fetch().await?;
        let mime = doc.mime_type();

        let result = self.client.predict_with_file("yolo_detect", bytes, "media", mime, vec![]).await?;
        let mut parsed = GradioClient::parse_result(result);

        if let Some(map) = parsed.as_object_mut() {
            if !map.contains_key("transcribed_text") {
                let mut summary = String::from("Detections: ");
                if let Some(detections) = map.get("detections").and_then(|v| v.as_array()) {
                    summary.push_str(&format!("{} items found.", detections.len()));
                } else {
                    summary.push_str("None.");
                }
                map.insert("transcribed_text".to_string(), serde_json::Value::String(summary));
            }
        }

        doc.update_extracted_information(parsed, db).await?;
        tracing::info!("[YoloProcessor][FINISH] Completed YOLO processing for doc_id={}", doc_id);
        Ok(())
    }

    async fn cron_func(&self, db: &DatabaseConnection, store: &S3ObjectStore) -> Result<(), DocumentErrors> {
        tracing::debug!("[YoloProcessor][CRON] Checking for unprocessed image documents...");
        
        // Note: The Video variant will be added by migration but may not yet be generated in the ORM. 
        // To handle this safely, we query Image documents only (Video will be processed via manual trigger from model_router).
        let documents = document::Entity::find()
            .filter(document::Column::Type.is_in([DocumentType::Image, DocumentType::Video]))
            .filter(document::Column::Status.eq(DocumentStatus::Success))
            .filter(document::Column::ExtractedInformation.is_null())
            .all(db)
            .await
            .map_err(|e| {
                tracing::error!("[YoloProcessor][DB_ERROR] Failed to query: {}", e);
                DocumentErrors::DatabaseError(e)
            })?;

        if documents.is_empty() {
            return Ok(());
        }
        tracing::info!("[YoloProcessor][CRON] Found {} unprocessed document(s)", documents.len());

        for model in documents {
            let doc_id = model.id;
            let mut active: document::ActiveModel = model.clone().into();
            active.status = Set(DocumentStatus::Processing);
            active.updated_at = Set(chrono::Utc::now().fixed_offset());
            if let Err(e) = active.update(db).await {
                tracing::error!("[YoloProcessor][DB_ERROR] Failed to set status=Processing for doc_id={}: {}", doc_id, e);
                continue;
            }

            let result = match model.r#type {
                DocumentType::Video => {
                    let video_doc = Video::new(doc_id, model.object_key.clone(), store.clone());
                    self.process(&video_doc, db).await
                }
                _ => {
                    let image_doc = Image::new(doc_id, model.object_key.clone(), store.clone());
                    self.process(&image_doc, db).await
                }
            };

            if let Err(e) = result {
                tracing::error!("[YoloProcessor][PROCESSING_ERROR] Error for doc_id={}: {}", doc_id, e);
            }

            if let Ok(Some(d)) = document::Entity::find_by_id(doc_id).one(db).await {
                let mut active: document::ActiveModel = d.into();
                active.status = Set(DocumentStatus::Finish);
                active.updated_at = Set(chrono::Utc::now().fixed_offset());
                if let Err(e) = active.update(db).await {
                    tracing::error!("[YoloProcessor][DB_ERROR] Failed to set status=Finish for doc_id={}: {}", doc_id, e);
                } else {
                    tracing::info!("[YoloProcessor][STATUS_TRANSITION] doc_id={} -> Finish", doc_id);
                }
            }
        }
        Ok(())
    }
}
