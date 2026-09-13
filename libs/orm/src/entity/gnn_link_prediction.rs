use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "gnn_link_prediction")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,
    pub target_entity_id: String,
    pub target_entity_type: String,
    pub candidate_entity_id: String,
    pub candidate_entity_type: String,
    pub predicted_edge_type: String,
    #[sea_orm(column_type = "Double")]
    pub link_probability: f64,
    pub is_hypothesis_flagged: bool,
    #[sea_orm(column_type = "Double", nullable)]
    pub confidence_threshold: Option<f64>,
    #[sea_orm(column_type = "Text", nullable)]
    pub recommendation: Option<String>,
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
