# Changelog
All notable changes to this project will be documented in this file.

## [0.9.0] - 2024-04-29
### Changed
- Remove the `Result` in the `run` API [#43](https://github.com/snipsco/gazetteer-entity-parser/pull/43)
- Consume the `Parser` object during injection [#43](https://github.com/snipsco/gazetteer-entity-parser/pull/43)
- Improve the memory footprint of `ResolvedSymbolTable` [#43](https://github.com/snipsco/gazetteer-entity-parser/pull/43)
- Update dependencies and remove examples + benches [#46](https://github.com/snipsco/gazetteer-entity-parser/pull/46)

## [0.8.0] - 2019-08-27
### Changed
- Add `max_alternatives` parameter to the `Parser::run` API [#39](https://github.com/snipsco/gazetteer-entity-parser/pull/39)
- Add `alternatives` attribute in `ParsedValue` [#39](https://github.com/snipsco/gazetteer-entity-parser/pull/39)
- Switch `matched_value` and `raw_value` [#39](https://github.com/snipsco/gazetteer-entity-parser/pull/39)
- Group `resolved_value` and `matched_value` in a dedicated `ResolvedValue` object [#39](https://github.com/snipsco/gazetteer-entity-parser/pull/39)

## [0.7.2] - 2019-07-19
### Fixed
- Make `LicenseInfo` public [#38](https://github.com/snipsco/gazetteer-entity-parser/pull/38)

## [0.7.1] - 2019-07-18
### Added
- Add a license file to the gazetteer entity parser [#36](https://github.com/snipsco/gazetteer-entity-parser/pull/36)

## [0.7.0] - 2019-04-16
### Added
- Add API to prepend entity values [#31](https://github.com/snipsco/gazetteer-entity-parser/pull/31)
- Add matched value in API output [#32](https://github.com/snipsco/gazetteer-entity-parser/pull/32)

## [0.6.0] - 2018-11-09
### Changed
- Optimize memory usage [#27](https://github.com/snipsco/gazetteer-entity-parser/pull/27)
- Simpler pattern for errors [#27](https://github.com/snipsco/gazetteer-entity-parser/pull/27)

## [0.5.1] - 2018-10-15
### Changed
- Fix bug affecting the backward expansion of possible matches starting with stop words [#25](https://github.com/snipsco/gazetteer-entity-parser/pull/25)

## [0.5.0] - 2018-10-01
### Changed
- Clearer `ParserBuilder`'s API 

[0.8.0]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.7.2...0.8.0
[0.7.2]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.7.1...0.7.2
[0.7.1]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.5.1...0.6.0
[0.5.1]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/snipsco/gazetteer-entity-parser/compare/0.4.2...0.5.0
