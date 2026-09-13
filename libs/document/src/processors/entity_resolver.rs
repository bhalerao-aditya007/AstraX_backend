use std::sync::Arc;
use async_trait::async_trait;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};

use crate::{
    document::base::Document,
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityResolveResult {
    pub name1: String,
    pub name2: String,
    pub merge_probability: f64,
    pub action: String,
}

pub struct EntityResolverProcessor {
    client: Arc<GradioClient>,
}

impl EntityResolverProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }

    pub async fn resolve_entities(
        &self,
        candidates: Vec<(String, String)>,
    ) -> Result<Vec<EntityResolveResult>, DocumentErrors> {
        tracing::info!("[EntityResolver] Resolving entities for {} candidates", candidates.len());
        let payload = serde_json::json!({ "candidates": candidates });
        let result = self.client.predict_json("entity_resolve", payload).await?;
        let parsed = GradioClient::parse_result(result);
        
        let results: Vec<EntityResolveResult> = serde_json::from_value(parsed)
            .map_err(|e| DocumentErrors::StorageError(format!("Failed to parse resolve results: {}", e)))?;
            
        Ok(results)
    }
}

#[async_trait]
impl DocumentProcessor for EntityResolverProcessor {
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
