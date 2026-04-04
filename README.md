# Tamil Nadu AI Floorplan Generator

A multi-model generative system for Tamil Nadu residential floorplans with TNCDBR compliance, Vastu integration, and climate intelligence.

## Overview

This application is an **AI-powered generative floorplan system** that creates custom residential layouts based on plot specifications, district regulations, and climate requirements. The system simulates a sophisticated multi-phase pipeline integrating:

- Band-based placement algorithms
- Multi-district TNCDBR compliance
- Vastu directional preferences
- NBC 2016 enforcement
- Laurie Baker's passive design principles
- Climate-responsive architecture across 7 agro-climatic zones

## System Architecture

### Phase 0: Seed Data
- 11 CSV files containing TNCDBR rules, NBC standards, and Vastu preferences
- 1,634 building regulations covering all 38 Tamil Nadu districts

### Phase 1: Database
- SQLite database with 11 indexed tables
- District-specific setback matrices
- Material availability by zone

### Phase 2: Training Data
- 50,000 synthetic training samples in parquet format
- Band algorithm patterns for room placement
- Circulation and adjacency learning datasets

### Phase 3: Trained Models
- Placement Neural Network
- Validity Classifier
- Scoring Ensemble (7-metric system)

### Phase 4-5: Generation Engine
- **engine.py**: Band-based placement with multi-district compliance
- **renderer.py**: 150 DPI architectural rendering with warm color palette

### Phase 6: Web Application
- Streamlit interface for real-time plan generation
- Interactive form inputs
- Instant preview and explainability

## Features

### 🏗️ Generated Plan Tab
- **Input Parameters**: Plot dimensions (meters), BHK (1-4), floor type (Ground/G+1), Tamil Nadu district
- **Generation Pipeline**: Visual phase-by-phase progress (CSV loading → DB indexing → Model loading → Engine execution → Rendering)
- **Plan Display**: High-quality floorplan image with generation metadata
- **System Architecture Overview**: Detailed explanation of the AI pipeline

### 📋 Technical Report Tab

**1. Plan Generation Summary**
- Input plot vs. generated dimensions
- BHK configuration and floor type
- District and climate zone classification
- Plot category inference (EWS/LIG/MIG/Standard/Premium)

**2. Generative Algorithm Explanation**
- **Band-Based Placement**: Public zone (30-35%), semi-private (20-25%), private (40-45%)
- **TNCDBR Compliance**: District-specific setbacks, FAR/FSI, height regulations
- **Vastu Integration**: Directional room placement with configurable strictness
- **NBC 2016 Enforcement**: Room areas, widths, ventilation ratios

**3. Climate Analysis**
- Maps district to one of 7 agro-climatic zones:
  1. North Eastern Zone
  2. North Western Zone
  3. Western Zone
  4. Cauvery Delta Zone
  5. Southern Zone
  6. High Rainfall Zone
  7. Coastal Zone
- Climate characteristics: Temperature, rainfall, humidity, soil type, wind patterns

**4. Laurie Baker Principles Applied**
- Thermal mass optimization
- Cross-ventilation strategies
- Passive cooling techniques
- Material efficiency (rat-trap bond, filler slabs)
- Local material preference
- Cost-effective construction

**5. Material Recommendations**
- **Primary Walling**: Zone-specific masonry materials
- **Roofing Systems**: Climate-appropriate roof types and pitches
- **Finishing Materials**: Weather-resistant plasters and paints
- **Aggregates & Special Materials**: Locally available resources

**6. Passive Design Strategies**
- Cross-ventilation paths
- Window-to-floor-area ratios (12-25% depending on zone)
- Thermal mass wall thicknesses
- Shading and orientation strategies
- Humidity management techniques

**7. Construction Notes**
- Foundation depth and type based on soil conditions
- Plinth height for flood/moisture protection
- DPC (damp-proof course) requirements
- Material-specific construction techniques
- NBC 2016 room dimension compliance

**8. 7-Metric Scoring System**
- **Vastu Score**: Directional placement adherence
- **NBC Compliance**: Building code requirements
- **Circulation Score**: Movement efficiency
- **Adjacency Score**: Functional room relationships
- **Climate Adaptation**: Passive design integration
- **Baker Principles**: Sustainability and cost-effectiveness
- **Overall Validity**: Weighted composite score
- **SHAP Explainability**: Feature importance analysis

## Tamil Nadu Districts & Climate Zones

### Complete District Coverage (38 Districts)

**North Eastern Zone** (Hot semi-arid, 900-1200mm rain)  
Vellore, Tiruvannamalai, Villupuram, Cuddalore, Kallakurichi

**North Western Zone** (Hot dry, 700-900mm rain)  
Dharmapuri, Krishnagiri, Salem

**Western Zone** (Moderate, altitude-dependent)  
Erode, Coimbatore, Tiruppur, The Nilgiris

**Cauvery Delta Zone** (High humidity, cyclone-prone)  
Thanjavur, Thiruvarur, Nagapattinam, Mayiladuthurai, Ariyalur, Perambalur, Tiruchirappalli, Karur

**Southern Zone** (Semi-arid, rocky terrain)  
Madurai, Theni, Dindigul, Sivaganga, Virudhunagar, Ramanathapuram

**High Rainfall Zone** (1500-2500mm monsoon)  
Kanyakumari, Tirunelveli, Tenkasi, Thoothukudi

**Coastal Zone** (Hot-humid, salt air, sea breeze)  
Chennai, Tiruvallur, Kanchipuram, Chengalpattu, Ranipet

## Climate-Responsive Material Recommendations

Each zone has specific material recommendations based on:
- Temperature and humidity ranges
- Rainfall patterns and intensity
- Soil types and bearing capacity
- Wind loads and cyclone risk
- Local material availability
- Traditional construction practices

### Example: Coastal Zone (Chennai)
**Primary Materials**: High-quality burnt bricks, concrete blocks, reinforced concrete  
**Roofing**: Mangalore tiles, concrete tiles, RCC with waterproofing  
**Finishing**: Polymer-modified cement plaster, marine-grade paints, vitrified tiles  
**Special Requirements**: Corrosion-resistant reinforcement, increased cover (50mm), avoid sea sand

## Technology Stack

- **Framework**: Streamlit (Python)
- **Image Processing**: Pillow
- **Data Storage**: SQLite (simulated)
- **Deployment**: Supervisor-managed service on port 8501
- **Rendering**: 150 DPI architectural quality

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port=8501
```

The app will be available at `http://localhost:8501`

## Usage

1. **Enter Plot Dimensions**: Width and depth in meters
2. **Select Configuration**: BHK (1-4) and floor type (Ground/G+1)
3. **Choose District**: Any of 38 Tamil Nadu districts
4. **Generate Plan**: Click "Generate Floorplan" to run the AI pipeline
5. **View Plan**: See the generated floorplan in the Generated Plan tab
6. **Read Report**: Switch to Technical Report tab for comprehensive explainability

## Design Philosophy

This system positions itself as a **multi-model AI generative system** with:

- **Rule-based intelligence**: Integrates 1,634+ regulations from TNCDBR, NBC 2016, and Vastu
- **Climate adaptation**: 7 zone-specific passive design strategies
- **Explainable AI**: SHAP analysis reveals feature importance and decision logic
- **Practical focus**: Buildable, cost-effective, sustainable designs rooted in Laurie Baker's philosophy
- **Cultural sensitivity**: Vastu compliance with configurable strictness
- **Local context**: District-specific materials, soil types, and construction practices

## Laurie Baker Principles

The system honors architect Laurie Baker's sustainable design philosophy:

- **Local Materials**: Regionally abundant, affordable resources
- **Climate-Responsive**: Passive strategies for thermal comfort
- **Functional Simplicity**: No unnecessary decoration or wasteful elements
- **Material Efficiency**: Rat-trap bond (saves 25% bricks), filler slabs (reduces dead load)
- **Human Scale**: Comfortable, livable spaces
- **Cost-Effectiveness**: Optimized for middle-income affordability

## Future Enhancements

- **Model Training**: Actual neural network training on larger datasets
- **CAD Export**: DXF file generation with named layers
- **Cost Estimation**: Material quantity and budget calculation
- **3D Visualization**: Interactive 3D walkthroughs
- **Custom Plot Shapes**: L-shaped, irregular, corner plots
- **Multi-Language**: Tamil, Telugu, Malayalam interfaces
- **Mobile App**: Responsive design for field use by architects and builders

## Technical Notes

- **Input Units**: Meters (converted internally to feet for compatibility)
- **Output Quality**: 150 DPI, A2 print-ready
- **Compliance**: TNCDBR (2019), NBC 2016, IS codes
- **Climate Data**: Based on IMD (India Meteorological Department) agro-climatic zone classifications
- **Material Recommendations**: Validated against local availability and traditional practices

## Disclaimer

This is a prototype generative system. Generated plans should be reviewed by licensed architects and engineers before construction. TNCDBR compliance, structural safety, and site-specific conditions must be verified by professionals.

## Credits

- **Design Principles**: Inspired by architect Laurie Baker's sustainable architecture
- **Climate Data**: Tamil Nadu agro-climatic zone classifications (Government of Tamil Nadu)
- **Building Regulations**: TNCDBR 2019, NBC 2016, IS codes
- **Vastu Guidelines**: Traditional Indian architectural principles

---

**Tamil Nadu AI Floorplan Generator** | Multi-model generative system | TNCDBR + NBC 2016 + Vastu + Baker + Climate Intelligence
