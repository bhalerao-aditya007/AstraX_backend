pub use sea_orm_migration::prelude::*;

mod m20220101_000001_create_table;
mod m20260903_120814_document;
mod m20260907_160048_gnn;
mod m20260913_000001_entities;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20220101_000001_create_table::Migration),
            Box::new(m20260903_120814_document::Migration),
            Box::new(m20260907_160048_gnn::Migration),
            Box::new(m20260913_000001_entities::Migration),
        ]
    }
}
