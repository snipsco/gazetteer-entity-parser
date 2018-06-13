#[derive(Debug)]
pub struct EntityValue {
    pub weight: f32,
    pub raw_value: String,  // TODO: rename to resolved value
    pub verbalized_value: String,  // TODO: rename to raw value
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
