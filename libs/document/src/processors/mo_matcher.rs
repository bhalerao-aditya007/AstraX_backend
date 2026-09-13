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

pub struct MoMatcherProcessor {
    client: Arc<GradioClient>,
}

impl MoMatcherProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }

    pub async fn match_case(
        &self,
        case_entities: serde_json::Value,
    ) -> Result<serde_json::Value, DocumentErrors> {
        tracing::info!("[MoMatcherProcessor] Matching MO for case entities");
        let result = self.client.predict_json("mo_match", case_entities).await?;
        let parsed = GradioClient::parse_result(result);
        Ok(parsed)
    }
}

#[async_trait]
impl DocumentProcessor for MoMatcherProcessor {
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
