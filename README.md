# Tamil Nadu Plotwise Floorplan Explorer

A geometric retrieval engine for Tamil Nadu residential plot configurations with climate intelligence and explainability.

## Overview

This application is a **geometry-based rule engine** that matches residential plot specifications to appropriate floorplan layouts. It combines deterministic retrieval logic with explainable AI principles, providing users with:

1. **Exact floorplan matching** based on plot dimensions, BHK requirements, and floor type
2. **Climate-responsive design recommendations** based on Tamil Nadu's 7 agro-climatic zones
3. **Laurie Baker architectural principles** for cost-effective, sustainable construction
4. **District-specific material recommendations** aligned with local availability and climate needs

## Features

### 🏗️ Floorplan Tab
- **Input Parameters**: Plot width, depth, BHK (1-4), floor type (Ground/G+1), district
- **Intelligent Matching**: Finds exact matches from 25 standardized layouts
- **Visual Progress**: Simulates geometric algorithm evaluation stages
- **Match Confidence**: Shows match accuracy and provides alternatives when exact match unavailable
- **Image Viewer**: High-quality floorplan visualization with metadata

### 📋 Report Tab
- **Layout Selection Reasoning**: Explains why a specific plan was chosen
- **Plot Category Classification**: EWS, LIG, MIG, Standard, or Premium
- **Climate Zone Analysis**: Maps district to one of 7 agro-climatic zones
- **Baker Design Principles**: Climate-responsive strategies (thermal mass, ventilation, shading)
- **Material Recommendations**: Zone-specific, locally available construction materials
- **Passive Design Strategies**: Natural cooling, cross-ventilation, orientation guidelines
- **Construction Notes**: Practical implementation considerations

## Dataset

The application includes **25 pre-designed floorplan layouts** covering:

- **Plot sizes**: 5×9 ft to 25×25 ft
- **Configurations**: 1 BHK to 4 BHK
- **Floor types**: Ground floor and G+1 (two-storey)
- **Formats**: High-resolution PNG images

### Available Layouts
```
1 BHK: 5x9, 6x9, 6x12
2 BHK: 7x12, 7.5x10, 7.5x12, 9x12 (Ground and G+1 variants)
3 BHK: 10x15, 10x20, 12x15, 12x18, 15x15, 18x12 (Ground and G+1 variants)
4 BHK: 15x20, 18x20, 20x15, 20x18, 20x25, 25x25 (Ground and G+1 variants)
```

## Tamil Nadu Agro-Climatic Zones

The app maps all 38 Tamil Nadu districts into 7 zones:

1. **North Eastern Zone**: Hot semi-arid, moderate rainfall
2. **North Western Zone**: Hot dry, extreme summer heat
3. **Western Zone**: Moderate climate, varied microclimates
4. **Cauvery Delta Zone**: High humidity, cyclone-prone
5. **Southern Zone**: Semi-arid, rocky terrain
6. **High Rainfall Zone**: Heavy monsoon (1500-2500mm)
7. **Coastal Zone**: Hot and humid, sea breeze influence

Each zone has specific:
- Climate characteristics
- Laurie Baker design principles
- Recommended materials
- Passive design strategies
- Construction notes

## Technology Stack

- **Framework**: Streamlit (Python web framework)
- **Image Processing**: Pillow (PIL)
- **Styling**: Custom CSS (CAD-inspired professional theme)
- **Architecture**: Single-page application with deterministic matching logic

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Usage

1. **Configure Plot**: Enter plot dimensions, BHK, floor type, and district in the sidebar
2. **Generate Layout**: Click "Generate Layout" button
3. **View Floorplan**: See the matched layout with metadata in the Floorplan tab
4. **Read Report**: Switch to Report tab for detailed climate and material analysis

## Design Philosophy

This tool positions itself as a **geometric algorithm system** rather than a simple image gallery:

- **Rule-based retrieval**: Deterministic matching based on plot geometry
- **Explainable decisions**: Every selection is justified with reasoning
- **Climate intelligence**: Goes beyond layout to provide contextual design wisdom
- **Practical focus**: Emphasizes buildable, cost-effective, sustainable design

The visual staging (progress bars, algorithm steps) reinforces the perception of computational geometry evaluation, while the backend maintains reliable, explainable retrieval logic.

## Laurie Baker Principles

The app honors architect Laurie Baker's philosophy:

- **Local materials**: Use regionally abundant, affordable resources
- **Climate-responsive design**: Work with nature, not against it
- **Functional simplicity**: No unnecessary decoration
- **Thermal comfort**: Natural ventilation and thermal mass
- **Cost-effectiveness**: Efficient material usage (rat-trap bond, filler slabs)
- **Human scale**: Comfortable, livable spaces

## Future Enhancements

- Add more floorplan layouts (50+ designs)
- Integrate actual generative AI for custom plan creation
- Support custom plot shapes (L-shaped, irregular)
- Add 3D visualization
- Include cost estimation based on materials and zone
- Export plans as AutoCAD DWG files
- Multi-language support (Tamil, Telugu, Malayalam)

## License

This project is designed for educational and demonstration purposes.

## Credits

- **Design Principles**: Inspired by architect Laurie Baker's sustainable architecture philosophy
- **Climate Data**: Based on Tamil Nadu agro-climatic zone classifications
- **Building Regulations**: Aligned with TNCDBR (Tamil Nadu Combined Development and Building Rules)

---

**Tamil Nadu Plotwise Floorplan Explorer** | Geometric retrieval + climate intelligence engine
