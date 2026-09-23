use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    document::{base::Document, transcribed::TranscribedDocument},
    errors::DocumentErrors,
    gradio::GradioClient,
    processors::{base::DocumentProcessor, ner::NerProcessor},
    storage::s3_object_store::S3ObjectStore,
};
use sea_orm::DatabaseConnection;
pub struct OcrProcessor {
    client: Arc<GradioClient>,
    ner_processor: Arc<NerProcessor>,
}

impl OcrProcessor {
    pub fn new(client: Arc<GradioClient>, ner_processor: Arc<NerProcessor>) -> Self {
        Self {
            client,
            ner_processor,
        }
    }
}

#[async_trait]
impl DocumentProcessor for OcrProcessor {
    async fn process(
        &self,
        doc: &dyn Document,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let doc_id = doc.id();
        tracing::info!("[OcrProcessor][START] Processing doc_id={}", doc_id);

        // 1. Fetch raw image bytes from storage
        let bytes = doc.fetch().await.map_err(|e| {
            tracing::error!(
                "[OcrProcessor][STORAGE_ERROR] Failed to fetch image bytes for doc_id={}: {}",
                doc_id,
                e
            );
            e
        })?;

        let mime = doc.mime_type();

        // 2. Call the local TrOCR model via Gradio (api_name="ocr_handwritten")
        tracing::info!(
            "[OcrProcessor][GRADIO] Dispatching 'ocr_handwritten' prediction for doc_id={}",
            doc_id
        );

        let result = self
            .client
            .predict_with_file("ocr_handwritten", bytes, "diary_page.jpg", mime, vec![])
            .await
            .map_err(|e| {
                tracing::error!(
                    "[OcrProcessor][GRADIO_ERROR] 'ocr_handwritten' call failed for doc_id={}: {}",
                    doc_id,
                    e
                );
                e
            })?;

        let parsed = GradioClient::parse_result(result);

        let transcribed_text = parsed
            .get("transcribed_text")
            .and_then(|t| t.as_str())
            .unwrap_or_default()
            .to_string();

        tracing::info!(
            "[OcrProcessor] Extracted {} chars of transcribed_text for doc_id={}",
            transcribed_text.len(),
            doc_id
        );

        // 3. Chain to NerProcessor using TranscribedDocument in-memory adapter
        let transcribed_doc = TranscribedDocument::new(doc_id, transcribed_text);
        tracing::info!(
            "[OcrProcessor][CHAIN] Forwarding transcribed text to NerProcessor for doc_id={}",
            doc_id
        );

        self.ner_processor.process(&transcribed_doc, db).await?;

        tracing::info!(
            "[OcrProcessor][FINISH] Completed OCR & NER pipeline for doc_id={}",
            doc_id
        );
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
