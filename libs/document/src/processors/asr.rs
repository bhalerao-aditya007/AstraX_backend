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
    document::{base::Document, voice::Voice, transcribed::TranscribedDocument},
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::{base::DocumentProcessor, ner::NerProcessor},
    storage::s3_object_store::S3ObjectStore,
};

pub struct AsrProcessor {
    client: Arc<GradioClient>,
    ner_processor: Arc<NerProcessor>,
}

impl AsrProcessor {
    pub fn new(client: Arc<GradioClient>, ner_processor: Arc<NerProcessor>) -> Self {
        Self { client, ner_processor }
    }
}

#[async_trait]
impl DocumentProcessor for AsrProcessor {
    async fn process(&self, doc: &dyn Document, db: &DatabaseConnection) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[AsrProcessor][START] Processing doc_id={}", doc_id);

        let bytes = doc.fetch().await?;
        let mime = doc.mime_type();

        let mut parsed = match self.client.predict_with_file("transcribe", bytes, "audio.wav", mime, vec![]).await {
            Ok(res) => {
                let p = GradioClient::parse_result(res);
                if p.get("transcribed_text").is_some() || p.get("transcript").is_some() {
                    p
                } else {
                    return Err(DocumentErrors::ProcessingError("ASR model returned empty results".to_string()));
                }
            }
            Err(e) => {
                tracing::error!("[AsrProcessor] Inference call error: {}", e);
                return Err(DocumentErrors::ProcessingError(format!("ASR inference error: {}", e)));
            }
        };

        let mut text_to_ner = String::new();
        if let Some(map) = parsed.as_object() {
            if let Some(t) = map.get("transcribed_text").or_else(|| map.get("transcript")).and_then(|v| v.as_str()) {
                text_to_ner = t.to_string();
            }
        }

        if !text_to_ner.is_empty() {
            tracing::info!("[AsrProcessor][CHAIN] Forwarding text to NerProcessor for doc_id={}", doc_id);
            let transcribed_doc = TranscribedDocument::new(doc_id, text_to_ner);
            self.ner_processor.process(&transcribed_doc, db).await?;

            if let Some(model) = document::Entity::find_by_id(doc_id).one(db).await.map_err(DocumentErrors::DatabaseError)? {
                if let Some(ner_json) = model.extracted_information {
                    if let (serde_json::Value::Object(parsed_map), serde_json::Value::Object(ner_map)) = (&mut parsed, ner_json) {
                        for (k, v) in ner_map {
                            parsed_map.insert(k, v);
                        }
                    }
                }
            }
        }

        doc.update_extracted_information(parsed, db).await?;
        tracing::info!("[AsrProcessor][FINISH] Completed ASR processing for doc_id={}", doc_id);
        Ok(())
    }

    async fn cron_func(&self, db: &DatabaseConnection, store: &S3ObjectStore) -> Result<(), DocumentErrors> {
        tracing::debug!("[AsrProcessor][CRON] Checking for unprocessed voice documents...");
        let documents = document::Entity::find()
            .filter(document::Column::Type.eq(DocumentType::Voice))
            .filter(document::Column::Status.eq(DocumentStatus::Success))
            .filter(document::Column::ExtractedInformation.is_null())
            .all(db)
            .await
            .map_err(|e| {
                tracing::error!("[AsrProcessor][DB_ERROR] Failed to query: {}", e);
                DocumentErrors::DatabaseError(e)
            })?;

        if documents.is_empty() {
            return Ok(());
        }
        tracing::info!("[AsrProcessor][CRON] Found {} unprocessed voice document(s)", documents.len());

        for model in documents {
            let doc_id = model.id;
            let mut active: document::ActiveModel = model.clone().into();
            active.status = Set(DocumentStatus::Processing);
            active.updated_at = Set(chrono::Utc::now().fixed_offset());
            if let Err(e) = active.update(db).await {
                tracing::error!("[AsrProcessor][DB_ERROR] Failed to set status=Processing for doc_id={}: {}", doc_id, e);
                continue;
            }

            let voice_doc = Voice::new(doc_id, model.object_key.clone(), store.clone());
            let process_res = self.process(&voice_doc, db).await;
            if let Err(ref e) = process_res {
                tracing::error!("[AsrProcessor][PROCESSING_ERROR] Error for doc_id={}: {}", doc_id, e);
            }

            if let Ok(Some(d)) = document::Entity::find_by_id(doc_id).one(db).await {
                let mut active: document::ActiveModel = d.into();
                active.status = Set(if process_res.is_ok() {
                    DocumentStatus::Finish
                } else {
                    DocumentStatus::Failed
                });
                active.updated_at = Set(chrono::Utc::now().fixed_offset());
                if let Err(e) = active.update(db).await {
                    tracing::error!("[AsrProcessor][DB_ERROR] Failed to set status for doc_id={}: {}", doc_id, e);
                } else {
                    tracing::info!("[AsrProcessor][STATUS_TRANSITION] doc_id={} -> Finish/Failed", doc_id);
                }
            }
        }
        Ok(())
    }
}
