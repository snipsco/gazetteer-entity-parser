#[macro_export]
macro_rules! gazetteer {
    ($(($raw:expr, $resolved:expr),)*) => {{
        let mut gazetteer = Gazetteer::default();
        $(
            gazetteer.add(EntityValue {
                raw_value: $raw.to_string(),
                resolved_value: $resolved.to_string(),
            });
        )*
        gazetteer
    }}
}
