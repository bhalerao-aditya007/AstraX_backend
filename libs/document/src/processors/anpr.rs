use std::sync::Arc;
use async_trait::async_trait;
use sea_orm::DatabaseConnection;
use crate::{
    document::base::Document,
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

pub struct AnprProcessor {
    client: Arc<GradioClient>,
}

impl AnprProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl DocumentProcessor for AnprProcessor {
    async fn process(&self, doc: &dyn Document, db: &DatabaseConnection) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[AnprProcessor][START] Processing doc_id={}", doc_id);

        let bytes = doc.fetch().await?;
        let mime = doc.mime_type();

        let mut parsed = match self.client.predict_with_file("anpr", bytes, "image.jpg", mime, vec![]).await {
            Ok(res) => {
                let p = GradioClient::parse_result(res);
                if p.get("plate_number").is_some() || p.get("plate_text").is_some() {
                    p
                } else {
                    return Err(DocumentErrors::ProcessingError("ANPR model returned empty results".to_string()));
                }
            }
            Err(e) => {
                tracing::error!("[AnprProcessor] Inference call error: {}", e);
                return Err(DocumentErrors::ProcessingError(format!("ANPR inference error: {}", e)));
            }
        };

        if let Some(map) = parsed.as_object_mut() {
            if !map.contains_key("transcribed_text") {
                if let Some(plate) = map.get("plate_number").or_else(|| map.get("plate_text")).and_then(|v| v.as_str()) {
                    map.insert("transcribed_text".to_string(), serde_json::Value::String(plate.to_string()));
                }
            }
        }

        doc.update_extracted_information(parsed, db).await?;
        tracing::info!("[AnprProcessor][FINISH] Completed ANPR for doc_id={}", doc_id);
        Ok(())
    }

    async fn cron_func(&self, _db: &DatabaseConnection, _store: &S3ObjectStore) -> Result<(), DocumentErrors> {
        tracing::debug!("[AnprProcessor][CRON] Skipping ANPR cron as it is triggered manually or by YOLO.");
        Ok(())
    }
}
