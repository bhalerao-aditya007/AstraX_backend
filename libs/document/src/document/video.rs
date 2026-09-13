use async_trait::async_trait;
use orm::entity::document;
use serde_json::Value;
use uuid::Uuid;

use crate::{document::base::Document, errors::DocumentErrors, storage::object_store::ObjectStore};
use sea_orm::{ActiveModelTrait, ActiveValue::Set, DatabaseConnection, EntityTrait};

pub struct Video<O: ObjectStore> {
    id: Uuid,
    key: String,
    storage: O,
}

impl<O: ObjectStore> Video<O> {
    pub fn new(id: Uuid, key: String, storage: O) -> Self {
        Self { id, key, storage }
    }
}

#[async_trait]
impl<O: ObjectStore> Document for Video<O> {
    fn id(&self) -> Uuid {
        self.id
    }

    fn mime_type(&self) -> &str {
        let lower = self.key.to_lowercase();
        if lower.ends_with(".avi") {
            "video/x-msvideo"
        } else if lower.ends_with(".mkv") {
            "video/x-matroska"
        } else if lower.ends_with(".mov") {
            "video/quicktime"
        } else if lower.ends_with(".webm") {
            "video/webm"
        } else {
            "video/mp4"
        }
    }

    async fn fetch(&self) -> Result<Vec<u8>, DocumentErrors> {
        self.storage.get(&self.key).await
    }

    async fn update_extracted_information(
        &self,
        json: Value,
        db: &DatabaseConnection,
    ) -> Result<(), DocumentErrors> {
        let result = document::Entity::find_by_id(self.id()).one(db).await;
        let result = result.map_err(DocumentErrors::DatabaseError)?;
        if result.is_none() {
            return Err(DocumentErrors::NotFound(String::from("did not found the video")));
        }
        let result = result.unwrap();
        let mut active_modal: document::ActiveModel = result.into();
        active_modal.extracted_information = Set(Some(json));
        active_modal.update(db).await.map_err(DocumentErrors::DatabaseError)?;

        Ok(())
    }
}
