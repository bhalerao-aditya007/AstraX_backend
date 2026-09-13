use std::sync::Arc;
use async_trait::async_trait;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    document::base::Document,
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::base::DocumentProcessor,
    storage::s3_object_store::S3ObjectStore,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryGenerationPayload {
    pub case_id: String,
    pub fact_sheet: serde_json::Value,
    pub graph_data: serde_json::Value,
    pub mo_matches: serde_json::Value,
    pub forensic_evidence: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryCard {
    pub theory_id: String,
    pub rank: i32,
    pub confidence_score: f64,
    pub title: String,
    pub motive_category: String,
    pub chronology: Vec<ChronologyStep>,
    pub unresolved_gaps: Vec<UnresolvedGap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronologyStep {
    pub step: i32,
    pub timestamp_window: String,
    pub action: String,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnresolvedGap {
    pub description: String,
    pub recommended_action: String,
}

pub struct TheoryProcessor {
    client: Arc<GradioClient>,
}

impl TheoryProcessor {
    pub fn new(client: Arc<GradioClient>) -> Self {
        Self { client }
    }

    pub async fn generate_theories(
        &self,
        payload: TheoryGenerationPayload,
    ) -> Result<Vec<TheoryCard>, DocumentErrors> {
        tracing::info!("[TheoryProcessor] Generating theories for case {}", payload.case_id);
        
        let payload_json = serde_json::to_value(payload)
            .map_err(|e| DocumentErrors::StorageError(format!("Serialization failed: {}", e)))?;
            
        let result = self.client.predict_json("generate_theory", payload_json).await?;
        let parsed = GradioClient::parse_result(result);
        
        if let Some(arr) = parsed.as_array() {
            let mut theories = Vec::new();
            let mut all_parsed = true;
            for val in arr {
                match serde_json::from_value::<TheoryCard>(val.clone()) {
                    Ok(theory) => theories.push(theory),
                    Err(_) => {
                        all_parsed = false;
                        break;
                    }
                }
            }
            if all_parsed && !theories.is_empty() {
                return Ok(theories);
            }
        }
        
        let fallback = TheoryCard {
            theory_id: uuid::Uuid::new_v4().to_string(),
            rank: 1,
            confidence_score: 1.0,
            title: parsed.to_string(),
            motive_category: "Unknown".to_string(),
            chronology: vec![],
            unresolved_gaps: vec![],
        };
        Ok(vec![fallback])
    }
}

#[async_trait]
impl DocumentProcessor for TheoryProcessor {
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
