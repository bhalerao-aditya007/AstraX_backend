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

pub struct FinancialProcessor {
    client: Arc<GradioClient>,
}

impl FinancialProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }
}

#[async_trait]
impl DocumentProcessor for FinancialProcessor {
    async fn process(&self, doc: &dyn Document, db: &DatabaseConnection) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[FinancialProcessor][START] Processing doc_id={}", doc_id);

        let bytes = doc.fetch().await?;
        let text_content = String::from_utf8_lossy(&bytes).to_string();

        let input_json = serde_json::json!({
            "text": text_content
        });

        let result = self.client.predict_json("financial_analyze", input_json).await?;
        let mut parsed = GradioClient::parse_result(result);

        if let Some(map) = parsed.as_object_mut() {
            if !map.contains_key("transcribed_text") {
                map.insert("transcribed_text".to_string(), serde_json::Value::String("Financial Analysis Completed".to_string()));
            }
        }

        doc.update_extracted_information(parsed, db).await?;
        tracing::info!("[FinancialProcessor][FINISH] Completed financial analysis for doc_id={}", doc_id);
        Ok(())
    }

    async fn cron_func(&self, _db: &DatabaseConnection, _store: &S3ObjectStore) -> Result<(), DocumentErrors> {
        tracing::debug!("[FinancialProcessor][CRON] Skipping financial cron as it conflicts with NER and is triggered manually.");
        Ok(())
    }
}
