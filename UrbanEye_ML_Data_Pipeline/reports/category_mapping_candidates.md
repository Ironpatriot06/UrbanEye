# Candidate category mappings

Generated 2026-09-14T18:10:41.052884+00:00

**142 unmapped source categories covering 358,838 rows.** Nothing below has been applied.

| Confidence | Source categories | Rows |
|---|---|---|
| high | 11 | 76,306 |
| medium | 53 | 60,036 |
| low | 28 | 51,151 |
| none | 50 | 171,345 |

## Top 40 by row count

| Source | Category | Rows | Proposal | Confidence | Shared tokens | Evidence |
|---|---|---|---|---|---|---|
| chicago311 | Alley Light Out Complaint | 42,315 | STREETLIGHT_FAULT | high | light out | resembles already-mapped chicago311:Street Light Out Complaint |
| chicago311 | Stray Animal Complaint | 17,666 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Fly Dumping Complaint | 16,272 | GARBAGE_DUMPING | medium | dumping | resembles already-mapped nyc311:Illegal Dumping |
| chicago311 | Check for Leak | 14,410 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Yard Waste Pick-Up Request | 13,764 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Recycling Pick Up | 13,171 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Report an Injured Animal | 13,055 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Inspect Public Way Request | 12,110 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Tree Planting Request | 12,049 | FALLEN_TREE | medium | tree | resembles already-mapped chicago311:Tree Emergency |
| chicago311 | Vacant/Abandoned Building Complaint | 11,491 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Business Complaints | 10,551 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Buildings - Plumbing Violation | 10,189 | GARBAGE_DUMPING | low | violation | resembles already-mapped chicago311:Sanitation Code Violation; weak overlap |
| chicago311 | Missed Garbage Pick-Up Complaint | 9,863 | GARBAGE_DUMPING | medium | garbage | resembles already-mapped smartathon:GARBAGE |
| chicago311 | Vicious Animal Complaint | 9,840 | — | none |  | no token overlap with any existing mapping |
| chicago311 | No Building Permit and Construction Violation | 8,442 | GARBAGE_DUMPING | low | violation | resembles already-mapped chicago311:Sanitation Code Violation; ambiguous — runner-up CONST |
| chicago311 | Clean Vacant Lot Request | 8,253 | FALLEN_TREE | low | clean | resembles already-mapped chicago311:Tree Debris Clean-Up Request; weak overlap |
| chicago311 | Sign Repair Request - Stop Sign | 8,210 | SIGNAGE_DAMAGE | high | repair sign | resembles already-mapped sf311:Sign Repair |
| chicago311 | Nuisance Animal Complaint | 7,504 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Water Lead Test Visit Request | 7,215 | WATER_LEAKAGE_WATERLOGGING | low | water | resembles already-mapped chicago311:Water in Basement Complaint; weak overlap |
| chicago311 | Cab Feedback | 7,133 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Snow – Uncleared Sidewalk Complaint | 6,921 | FOOTPATH_DAMAGE | medium | sidewalk | resembles already-mapped chicago311:Sidewalk Inspection Request |
| chicago311 | Street Light Pole Damage Complaint | 6,869 | STREETLIGHT_FAULT | high | light street | resembles already-mapped chicago311:Street Light Out Complaint |
| chicago311 | Restaurant Complaint | 6,120 | — | none |  | no token overlap with any existing mapping |
| sf311 | Homeless Concerns | 5,205 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Street Light On During Day Complaint | 4,827 | STREETLIGHT_FAULT | high | light street | resembles already-mapped chicago311:Street Light Out Complaint |
| chicago311 | No Water Complaint | 4,751 | WATER_LEAKAGE_WATERLOGGING | high | water | resembles already-mapped chicago311:Water in Basement Complaint |
| chicago311 | Pet Wellness Check Request | 4,463 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Sign Repair Request - One Way Sign | 4,319 | SIGNAGE_DAMAGE | high | repair sign | resembles already-mapped sf311:Sign Repair |
| chicago311 | Low Water Pressure Complaint | 3,384 | WATER_LEAKAGE_WATERLOGGING | medium | water | resembles already-mapped chicago311:Water in Basement Complaint |
| chicago311 | Ice and Snow Removal Request | 3,250 | GRAFFITI_VISUAL_POLLUTION | low | removal | resembles already-mapped chicago311:Graffiti Removal Request; ambiguous — runner-up FALLEN |
| chicago311 | Alley Sewer Inspection Request | 3,121 | DRAINAGE_SEWER | low | sewer | resembles already-mapped sf311:Sewer Issues; ambiguous — runner-up POTHOLE scores 0.42 vs  |
| chicago311 | Consumer Fraud Complaint | 3,082 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Coyote Interaction Complaint | 2,963 | — | none |  | no token overlap with any existing mapping |
| chicago311 | Shared Cost Sidewalk Program Request | 2,929 | FOOTPATH_DAMAGE | medium | sidewalk | resembles already-mapped chicago311:Sidewalk Inspection Request |
| chicago311 | Vehicle Parked in Bike Lane Complaint | 2,886 | — | none |  | no token overlap with any existing mapping |
| sf311 | Sidewalk or Curb | 2,375 | FOOTPATH_DAMAGE | high | sidewalk | resembles already-mapped chicago311:Sidewalk Inspection Request |
| sf311 | Temporary Sign Request | 2,340 | SIGNAGE_DAMAGE | medium | sign | resembles already-mapped sf311:Sign Repair |
| chicago311 | City Vehicle Sticker Violation | 2,283 | GARBAGE_DUMPING | low | violation | resembles already-mapped chicago311:Sanitation Code Violation; weak overlap |
| sf311 | Noise Report | 2,095 | — | none |  | no token overlap with any existing mapping |
| sf311 | Blocked Street or SideWalk | 2,084 | ROAD_DAMAGE | low | street | resembles already-mapped nyc311:Street Condition; ambiguous — runner-up FOOTPATH_DAMAGE sc |

## How to act on this

1. Read the `high` rows first — they are the cheapest wins and the least ambiguous.
2. Append accepted rows to `config/category_mapping.csv` with a `notes` field saying who accepted them and why.
3. Re-run the pipeline. The category prior and the density features both change, so re-read `reports/statistical_audit.md` afterwards.
4. Anything genuinely ambiguous belongs in `REVIEW_REQUIRED`, not in a canonical category — that sentinel exists precisely so a guess is never necessary.
