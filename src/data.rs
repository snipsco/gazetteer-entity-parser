#[derive(Debug)]
pub struct EntityValue {
    pub weight: f32,
    pub resolved_value: String,
    pub raw_value: String
}

#[derive(Debug)]
pub struct Gazetteer {
    pub data: Vec<EntityValue>,
}

impl Gazetteer {
    pub fn add(&mut self, value: EntityValue) {
        self.data.push(value);
    }
}
