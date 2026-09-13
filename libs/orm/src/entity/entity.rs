use super::sea_orm_active_enums::EntityType;
use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "entity")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub entity_type: EntityType,
    pub canonical_id: String,
    pub display_name: String,
    #[sea_orm(column_type = "Text")]
    pub raw_value: String,
    #[sea_orm(column_type = "Double")]
    pub confidence: f64,
    pub source_document_id: Option<Uuid>,
    pub case_id: Option<Uuid>,
    pub metadata: Option<Json>,
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::document::Entity",
        from = "Column::SourceDocumentId",
        to = "super::document::Column::Id",
        on_update = "NoAction",
        on_delete = "SetNull"
    )]
    Document,
    #[sea_orm(
        belongs_to = "super::case::Entity",
        from = "Column::CaseId",
        to = "super::case::Column::Id",
        on_update = "NoAction",
        on_delete = "Cascade"
    )]
    Case,
}

impl Related<super::document::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Document.def()
    }
}

impl Related<super::case::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Case.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
