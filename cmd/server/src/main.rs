use std::sync::Arc;

use document::{
    document_manager::DocumentManager,
    gradio::GradioClient,
    processors::{
        asr::AsrProcessor,
        base::DocumentProcessor,
        fir_ocr::FirOcrProcessor,
        ner::NerProcessor,
        yolo::YoloProcessor,
    },
    storage::s3_object_store::S3ObjectStore,
};
use sea_orm::Database;

mod router;

#[tokio::main]
async fn main() {
    // Load .env
    let _ = dotenvy::dotenv();

    // Init tracing with environment filter
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,server=debug,document=debug,tower_http=info".into());
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .init();

    // Read config from env
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let s3_bucket = std::env::var("S3_BUCKET_NAME").expect("S3_BUCKET_NAME must be set");
    let s3_region = std::env::var("S3_REGION").expect("S3_REGION must be set");
    let s3_endpoint = std::env::var("S3_ENDPOINT").expect("S3_ENDPOINT must be set");
    let s3_access_key = std::env::var("S3_ACCESS_KEY").expect("S3_ACCESS_KEY must be set");
    let s3_secret_key = std::env::var("S3_SECRET_KEY").expect("S3_SECRET_KEY must be set");
    let server_host = std::env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let server_port = std::env::var("PORT")
        .or_else(|_| std::env::var("SERVER_PORT"))
        .unwrap_or_else(|_| "8000".to_string());

    // Connect to database
    tracing::info!("Connecting to database...");
    let db = Database::connect(&database_url)
        .await
        .expect("Failed to connect to database");
    tracing::info!("Database connected successfully");

    // Create S3 object store
    let storage = S3ObjectStore::new(
        s3_bucket,
        s3_region,
        &s3_endpoint,
        s3_access_key,
        s3_secret_key,
    );

    // Create document manager
    let manager = Arc::new(DocumentManager::new(db.clone(), storage.clone()));

    // -----------------------------------------------------------------------
    // Background Processor Cron Configuration
    // -----------------------------------------------------------------------
    let enable_cron: bool = std::env::var("ENABLE_DOCUMENT_PROCESSOR_CRON")
        .map(|v| {
            let lower = v.trim().to_lowercase();
            lower == "true" || lower == "1" || lower == "yes"
        })
        .unwrap_or(true);

    if !enable_cron {
        tracing::info!("Document processor background cron is DISABLED (ENABLE_DOCUMENT_PROCESSOR_CRON=false).");
    } else {
        let cron_interval_secs: u64 = std::env::var("DOCUMENT_CRON_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);

        // Shared GradioClient for HF Space models
        let gradio_client = Arc::new(GradioClient::from_env());
        tracing::info!(
            "GradioClient configured for HF Space: {}",
            gradio_client.base_url()
        );

        // Shared NER processor (used by FIR OCR and ASR as downstream chain)
        let ner_processor = Arc::new(NerProcessor::from_env());

        // -------------------------------------------------------------------
        // Cron 1: NVIDIA NIM Universal Processor (Images, Text, Voice)
        //
        // This is the primary catch-all processor that handles documents
        // which don't have a specialized processor. It runs on all document
        // types with Status::Success and no extracted_information.
        // -------------------------------------------------------------------
        match document::processors::nvidia::NvidiaKimiProcessor::from_env() {
            Ok(nvidia_processor) => {
                let processor = Arc::new(nvidia_processor);
                let nvidia_db = db.clone();
                let nvidia_store = storage.clone();
                tracing::info!(
                    "Starting NVIDIA document processor cron (model: '{}', interval: {}s)...",
                    processor.model(),
                    cron_interval_secs
                );

                tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(std::time::Duration::from_secs(cron_interval_secs));
                    loop {
                        interval.tick().await;
                        if let Err(e) = processor.cron_func(&nvidia_db, &nvidia_store).await {
                            tracing::error!("[CRON][NVIDIA] Processing error: {}", e);
                        }
                    }
                });
            }
            Err(e) => {
                tracing::warn!(
                    "NVIDIA_API_KEY not set; NVIDIA processor cron disabled: {}",
                    e
                );
            }
        }

        // -------------------------------------------------------------------
        // Cron 2: FIR OCR Processor (Scanned FIR images → structured IIF-1)
        //
        // Runs alongside NVIDIA but targets FIR-specific image extraction.
        // Since both NVIDIA and FIR OCR cron query Image docs without
        // extracted_information, NVIDIA runs first (lower interval). FIR OCR
        // can be triggered manually via /api/models/fir-ocr/{doc_id}.
        // In cron mode, it processes images that NVIDIA has already partially
        // processed but need FIR-specific extraction.
        // -------------------------------------------------------------------
        {
            let fir_db = db.clone();
            let fir_store = storage.clone();
            let fir_processor = Arc::new(FirOcrProcessor::new(
                gradio_client.clone(),
                ner_processor.clone(),
            ));
            let fir_interval = cron_interval_secs * 2; // Run at half the frequency

            tracing::info!(
                "Starting FIR OCR processor cron (interval: {}s)...",
                fir_interval
            );

            tokio::spawn(async move {
                // Initial delay to let NVIDIA process first
                tokio::time::sleep(std::time::Duration::from_secs(15)).await;
                let mut interval =
                    tokio::time::interval(std::time::Duration::from_secs(fir_interval));
                loop {
                    interval.tick().await;
                    if let Err(e) = fir_processor.cron_func(&fir_db, &fir_store).await {
                        tracing::error!("[CRON][FIR_OCR] Processing error: {}", e);
                    }
                }
            });
        }

        // -------------------------------------------------------------------
        // Cron 3: ASR Processor (Audio → Transcript → NER)
        //
        // Processes Voice documents through Whisper ASR then chains to NER.
        // -------------------------------------------------------------------
        {
            let asr_db = db.clone();
            let asr_store = storage.clone();
            let asr_processor = Arc::new(AsrProcessor::new(
                gradio_client.clone(),
                ner_processor.clone(),
            ));

            tracing::info!(
                "Starting ASR processor cron (interval: {}s)...",
                cron_interval_secs
            );

            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(std::time::Duration::from_secs(cron_interval_secs));
                loop {
                    interval.tick().await;
                    if let Err(e) = asr_processor.cron_func(&asr_db, &asr_store).await {
                        tracing::error!("[CRON][ASR] Processing error: {}", e);
                    }
                }
            });
        }

        // -------------------------------------------------------------------
        // Cron 4: YOLO Detection Processor (Images → Object Detection)
        //
        // Detects persons, vehicles, and objects in image documents.
        // -------------------------------------------------------------------
        {
            let yolo_db = db.clone();
            let yolo_store = storage.clone();
            let yolo_processor = Arc::new(YoloProcessor::new(gradio_client.clone()));

            tracing::info!(
                "Starting YOLO detection processor cron (interval: {}s)...",
                cron_interval_secs
            );

            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(std::time::Duration::from_secs(cron_interval_secs));
                loop {
                    interval.tick().await;
                    if let Err(e) = yolo_processor.cron_func(&yolo_db, &yolo_store).await {
                        tracing::error!("[CRON][YOLO] Processing error: {}", e);
                    }
                }
            });
        }

        // -------------------------------------------------------------------
        // Cron 5: GNN Auto-Trigger
        //
        // Periodically runs GNN collective inference if GNN_AUTO_TRIGGER=true.
        // This connects new entities to historical criminal network data.
        // -------------------------------------------------------------------
        let gnn_auto_trigger: bool = std::env::var("GNN_AUTO_TRIGGER")
            .map(|v| {
                let lower = v.trim().to_lowercase();
                lower == "true" || lower == "1" || lower == "yes"
            })
            .unwrap_or(false);

        if gnn_auto_trigger {
            let gnn_db = db.clone();
            let gnn_store = storage.clone();
            let gnn_url = std::env::var("GNN_SERVICE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8001".to_string());
            let gnn_interval = cron_interval_secs * 4; // Run at quarter frequency

            tracing::info!(
                "Starting GNN auto-trigger cron (url: {}, interval: {}s)...",
                gnn_url,
                gnn_interval
            );

            tokio::spawn(async move {
                let processor =
                    document::processors::gnn::GnnProcessor::new(gnn_url);
                // Initial delay to let extraction processors populate data first
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                let mut interval =
                    tokio::time::interval(std::time::Duration::from_secs(gnn_interval));
                loop {
                    interval.tick().await;
                    if let Err(e) = processor.cron_func(&gnn_db, &gnn_store).await {
                        tracing::error!("[CRON][GNN] Inference error: {}", e);
                    }
                }
            });
        } else {
            tracing::info!("GNN auto-trigger is DISABLED. Use POST /api/gnn/trigger for manual inference.");
        }
    }

    // -----------------------------------------------------------------------
    // Build router and start server
    // -----------------------------------------------------------------------
    let app = router::create_router(manager);

    let addr = format!("{}:{}", server_host, server_port);
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind to address");

    axum::serve(listener, app).await.expect("Server error");
}
