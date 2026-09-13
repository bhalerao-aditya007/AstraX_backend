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

pub struct SummarizerProcessor {
    client: Arc<GradioClient>,
}

impl SummarizerProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }

    pub async fn summarize_case(
        &self,
        case_data: serde_json::Value,
    ) -> Result<serde_json::Value, DocumentErrors> {
        tracing::info!("[Summarizer] Summarizing case data");
        let result = self.client.predict_json("summarize", case_data).await?;
        let parsed = GradioClient::parse_result(result);
        Ok(parsed)
    }
}

#[async_trait]
impl DocumentProcessor for SummarizerProcessor {
    async fn process(
        &self,
        _doc: &dyn Document,
        _db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        Ok(())
    }

    async fn cron_func(
        &self,
        _db: &DatabaseConnection,
        _store: &S3ObjectStore,
    ) -> Result<(), DocumentErrors> {
        Ok(())
    }
}
