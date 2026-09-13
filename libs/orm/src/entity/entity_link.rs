use super::sea_orm_active_enums::LinkSource;
use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "entity_link")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub source_entity_id: Uuid,
    pub target_entity_id: Uuid,
    pub link_type: String,
    pub source: LinkSource,
    #[sea_orm(column_type = "Double")]
    pub probability: f64,
    #[sea_orm(column_type = "Text", nullable)]
    pub evidence_summary: Option<String>,
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::entity::Entity",
        from = "Column::SourceEntityId",
        to = "super::entity::Column::Id",
        on_update = "NoAction",
        on_delete = "Cascade"
    )]
    SourceEntity,
    #[sea_orm(
        belongs_to = "super::entity::Entity",
        from = "Column::TargetEntityId",
        to = "super::entity::Column::Id",
        on_update = "NoAction",
        on_delete = "Cascade"
    )]
    TargetEntity,
}

impl ActiveModelBehavior for ActiveModel {}
