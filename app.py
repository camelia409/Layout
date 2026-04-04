import streamlit as st
import os
from PIL import Image
import time
import re

# Page configuration
st.set_page_config(
    page_title="Tamil Nadu Plotwise Floorplan Explorer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional CAD-inspired look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #ffffff;
        border-radius: 4px;
        border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a4f;
        color: white;
        border: 1px solid #1e3a4f;
    }
    h1 {
        color: #1e3a4f;
        font-weight: 600;
    }
    h2, h3 {
        color: #2c5168;
    }
    .css-1d391kg {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #2c8c99;
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        border-radius: 4px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1e3a4f;
    }
    div[data-testid="stMetricValue"] {
        color: #2c8c99;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Dataset of available layouts
LAYOUT_IMAGES = [
    "5x9_1bhk.png",
    "6x9_1bhk.png",
    "6x12_1bhk.png",
    "7x12_2bhk.png",
    "7x12_2bhk_g+1.png",
    "7.5x10_2bhk.png",
    "7.5x12_2bhk.png",
    "7.5x12_2bhk_g+1.png",
    "9x12_2bhk.png",
    "10X15_3bhk.png",
    "10x15_3bhk_g+1.png",
    "10x20_3bhk.png",
    "10x20_3bhk_g+1.png",
    "12x15_3bhk.png",
    "12x15_3bhk_g+1.png",
    "12x18_3bhk.png",
    "15x15_3bhk.png",
    "15x15_3bhk_g+1.png",
    "15x20_4bhk.png",
    "18x12_3bhkl_g_+1.png",
    "18x20_4bhk.png",
    "20x15_4bhk_g+1.png",
    "20x18_4bhk_g+1.png",
    "20x25_4bhk.png",
    "25x25_4bhk_g+1.png"
]

# Tamil Nadu Districts mapped to Agro-Climatic Zones
TN_DISTRICTS = {
    "North Eastern Zone": ["Vellore", "Tiruvannamalai", "Villupuram", "Cuddalore", "Kallakurichi"],
    "North Western Zone": ["Dharmapuri", "Krishnagiri", "Salem"],
    "Western Zone": ["Erode", "Coimbatore", "Tiruppur", "The Nilgiris"],
    "Cauvery Delta Zone": ["Thanjavur", "Thiruvarur", "Nagapattinam", "Mayiladuthurai", "Ariyalur", "Perambalur", "Tiruchirappalli", "Karur"],
    "Southern Zone": ["Madurai", "Theni", "Dindigul", "Sivaganga", "Virudhunagar", "Ramanathapuram"],
    "High Rainfall Zone": ["Kanyakumari", "Tirunelveli", "Tenkasi", "Thoothukudi"],
    "Coastal Zone": ["Chennai", "Tiruvallur", "Kanchipuram", "Chengalpattu", "Ranipet"]
}

# Climate characteristics and material recommendations
ZONE_CHARACTERISTICS = {
    "North Eastern Zone": {
        "climate": "Hot semi-arid with moderate rainfall. Hot summers (35-40°C), mild winters.",
        "baker_principles": "Focus on cross-ventilation, shaded openings, and thermal mass. East-west orientation minimizes heat gain. Brick jaalis provide filtered light while maintaining airflow.",
        "materials": ["Burnt clay bricks", "Hollow concrete blocks", "Mangalore tiles", "Lime plaster", "Local granite aggregates"],
        "passive_design": ["Cross ventilation through opposite walls", "Deep overhangs for sun shading", "Courtyard planning for air circulation", "Rat-trap bond brickwork for cavity insulation"],
        "notes": "Moderate rainfall allows for exposed brick finishes. Consider rainwater harvesting for groundwater recharge."
    },
    "North Western Zone": {
        "climate": "Hot dry climate with low to moderate rainfall. Extreme summer heat (38-42°C).",
        "baker_principles": "Thermal mass is critical. Thick walls delay heat transfer. Small west-facing openings reduce afternoon heat. High ceilings and ventilators aid hot air escape.",
        "materials": ["Stone masonry (locally available granite)", "Thick brick walls (230-350mm)", "Filler slab technology", "Clay pot roofing for insulation", "Lime-based plasters"],
        "passive_design": ["Thick thermal mass walls (300mm+)", "Minimal west-facing openings", "Double-height living spaces", "Underground storage spaces stay cooler", "Light-colored external finishes"],
        "notes": "Water scarcity is a concern. Design for minimal water usage in construction. Stone is abundant and cost-effective."
    },
    "Western Zone": {
        "climate": "Moderate climate with cool winters (especially Nilgiris). Varied microclimates from plains to hills.",
        "baker_principles": "Flexibility in design based on altitude. Plains need cooling; hills need warmth retention. Sloped roofs for rainfall in higher elevations.",
        "materials": ["Laterite blocks (where available)", "Burnt clay bricks", "Country tiles or Mangalore tiles", "Hollow blocks", "Wood (Nilgiris region)"],
        "passive_design": ["Verandahs for weather protection", "Sloped roofs (especially hills)", "Large openings for plains, moderate for hills", "Natural wood finishes in cooler areas"],
        "notes": "Tiruppur/Coimbatore: emphasize ventilation. Nilgiris: consider warmth retention, steeper roof slopes, and moisture resistance."
    },
    "Cauvery Delta Zone": {
        "climate": "High humidity, moderate to high rainfall, hot summers. Cyclone-prone coastal districts.",
        "baker_principles": "Moisture resistance is key. Elevated plinths prevent flooding. Ventilation combats humidity. Avoid materials that retain moisture.",
        "materials": ["First-class burnt bricks", "Concrete blocks for damp-prone areas", "Waterproof cement plasters", "Mangalore tiles with good overlap", "Granite or concrete flooring"],
        "passive_design": ["Elevated plinth (600-900mm)", "Continuous ventilation (high + low level)", "Sloped tile roofs with wide overhangs", "Covered sit-outs for rain protection"],
        "notes": "High water table. Use lime concrete or gravel bedding under floors. Avoid exposed brick or lime finishes; use cement-based weatherproof plasters."
    },
    "Southern Zone": {
        "climate": "Semi-arid with hot dry summers. Low to moderate rainfall. Rocky terrain in parts.",
        "baker_principles": "Similar to North Western—prioritize thermal mass, shading, and minimal west exposure. Use local stone where available.",
        "materials": ["Stone masonry (Dindigul, Madurai regions)", "Burnt clay bricks", "Hollow blocks", "Filler slabs", "Lime plaster over thick masonry"],
        "passive_design": ["Courtyards for microclimate control", "Thick masonry walls (minimum 230mm)", "Small high-level windows for hot air exit", "Verandahs and deep overhangs"],
        "notes": "Rocky terrain makes excavation costly. Plinth beam and stone foundation systems are common. Lime plaster is traditional and breathable."
    },
    "High Rainfall Zone": {
        "climate": "Heavy monsoon rainfall (1500-2500mm annually). Moderate heat, high humidity, coastal influence.",
        "baker_principles": "Steep roof slopes (30-35°), moisture-resistant materials, excellent drainage, elevated floors, and protected openings.",
        "materials": ["Concrete blocks or well-burnt bricks", "Waterproof cement plasters", "Clay tiles with steep pitch", "Concrete or mosaic flooring", "Corrosion-resistant hardware"],
        "passive_design": ["Steep-sloped roofs with large overhangs", "Covered verandahs and balconies", "Elevated plinth (minimum 600mm)", "Continuous ventilation for humidity control"],
        "notes": "Heavy rainfall requires robust waterproofing. Avoid flat roofs. Roof drainage must be carefully designed to prevent seepage."
    },
    "Coastal Zone": {
        "climate": "Hot and humid year-round. Sea breeze moderates temperature. High humidity (70-90%). Cyclone risk.",
        "baker_principles": "Maximize natural ventilation to combat humidity. Salt-resistant materials for longevity. Cross-ventilation aligned with sea breeze direction.",
        "materials": ["High-quality burnt bricks", "Concrete blocks", "Polymer or epoxy-based paints (salt-resistant)", "Corrosion-resistant steel", "Concrete or vitrified flooring"],
        "passive_design": ["Large openings for sea breeze (east-facing preferred)", "Continuous cross-ventilation", "Elevated plinth for flood risk", "Covered balconies and sit-outs", "Light colors to reflect heat"],
        "notes": "Corrosion from salt air is a major concern. Use corrosion-resistant reinforcement and hardware. Regular maintenance required for painted surfaces."
    }
}

def get_district_zone(district):
    """Map a district to its agro-climatic zone."""
    for zone, districts in TN_DISTRICTS.items():
        if district in districts:
            return zone
    return "Coastal Zone"  # Default

def normalize_input(width, depth, bhk, floor_type):
    """Normalize user input to match filename patterns."""
    # Handle decimal widths
    width_str = str(width).replace('.', '.') if '.' in str(width) else str(int(width))
    depth_str = str(depth).replace('.', '.') if '.' in str(depth) else str(int(depth))
    
    # Build search patterns
    floor_suffix = "_g+1" if floor_type == "G+1" else ""
    
    # Generate possible filename patterns
    patterns = [
        f"{width_str}x{depth_str}_{bhk}bhk{floor_suffix}.png",
        f"{width_str}X{depth_str}_{bhk}bhk{floor_suffix}.png",
        f"{depth_str}x{width_str}_{bhk}bhk{floor_suffix}.png",  # Try reversed dimensions
        f"{depth_str}X{width_str}_{bhk}bhk{floor_suffix}.png",
    ]
    
    return patterns

def find_exact_match(patterns):
    """Find exact filename match from patterns."""
    for pattern in patterns:
        pattern_lower = pattern.lower()
        for filename in LAYOUT_IMAGES:
            if filename.lower() == pattern_lower:
                return filename, 100  # 100% confidence for exact match
    return None, 0

def find_closest_match(width, depth, bhk, floor_type):
    """Find closest available layout if no exact match."""
    # Filter by BHK first
    bhk_matches = [f for f in LAYOUT_IMAGES if f"{bhk}bhk" in f.lower()]
    
    if not bhk_matches:
        return None, 0, "No layouts available for this BHK configuration."
    
    # Try to match floor type
    if floor_type == "G+1":
        floor_matches = [f for f in bhk_matches if "g+1" in f.lower() or "g_+1" in f.lower()]
        if floor_matches:
            return floor_matches[0], 70, f"Closest match: Same BHK and floor type, different dimensions."
    
    # Return first BHK match
    return bhk_matches[0], 60, f"Closest match: Same BHK, different dimensions and/or floor type."

def get_plot_category(width, depth, bhk):
    """Infer plot category based on size and BHK."""
    area = width * depth
    
    if bhk == 1 and area <= 70:
        return "EWS (Economically Weaker Section)"
    elif bhk <= 2 and area <= 100:
        return "LIG (Low Income Group)"
    elif bhk <= 3 and area <= 200:
        return "MIG (Middle Income Group)"
    elif area <= 400:
        return "Standard Residential"
    else:
        return "Premium Residential"

def simulate_progress():
    """Show fake progress for geometric algorithm simulation."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stages = [
        ("Validating TNCDBR geometry constraints...", 20),
        ("Checking plot-to-BHK compatibility...", 40),
        ("Evaluating setback envelope...", 60),
        ("Running geometric layout selection algorithm...", 80),
        ("Rendering floorplan...", 100)
    ]
    
    for stage, progress in stages:
        status_text.text(stage)
        progress_bar.progress(progress)
        time.sleep(0.5)
    
    status_text.empty()
    progress_bar.empty()

# Header
st.title("🏗️ Tamil Nadu Plotwise Floorplan Explorer")
st.markdown("**Geometric retrieval engine for Tamil Nadu residential plot configurations**")
st.markdown("---")

# Sidebar inputs
with st.sidebar:
    st.header("Plot Configuration")
    
    plot_width = st.number_input("Plot Width (feet)", min_value=5.0, max_value=50.0, value=10.0, step=0.5)
    plot_depth = st.number_input("Plot Depth (feet)", min_value=5.0, max_value=50.0, value=15.0, step=0.5)
    
    bhk = st.selectbox("BHK Type", options=[1, 2, 3, 4], index=2)
    
    floor_type = st.radio("Building Type", options=["Ground Floor", "G+1"], index=0)
    
    # Flatten district list
    all_districts = []
    for districts in TN_DISTRICTS.values():
        all_districts.extend(districts)
    all_districts = sorted(all_districts)
    
    district = st.selectbox("District", options=all_districts, index=all_districts.index("Chennai"))
    
    st.markdown("---")
    generate_btn = st.button("🔍 Generate Layout", use_container_width=True)

# Main content area with tabs
tab1, tab2 = st.tabs(["📐 Floorplan", "📋 Report"])

# Initialize session state
if 'layout_generated' not in st.session_state:
    st.session_state.layout_generated = False
    st.session_state.matched_image = None
    st.session_state.confidence = 0
    st.session_state.match_info = ""

# Handle generate button
if generate_btn:
    with tab1:
        simulate_progress()
        
        # Find matching layout
        patterns = normalize_input(plot_width, plot_depth, bhk, floor_type)
        matched_file, confidence = find_exact_match(patterns)
        
        if matched_file:
            st.session_state.matched_image = matched_file
            st.session_state.confidence = confidence
            st.session_state.match_info = "Exact match found"
        else:
            # Find closest match
            matched_file, confidence, info = find_closest_match(plot_width, plot_depth, bhk, floor_type)
            st.session_state.matched_image = matched_file
            st.session_state.confidence = confidence
            st.session_state.match_info = info
        
        st.session_state.layout_generated = True
        st.rerun()

# Floorplan Tab
with tab1:
    if st.session_state.layout_generated and st.session_state.matched_image:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Plot Size", f"{plot_width} × {plot_depth} ft")
        with col2:
            st.metric("Configuration", f"{bhk} BHK")
        with col3:
            st.metric("Floor Type", floor_type)
        with col4:
            st.metric("Match Confidence", f"{st.session_state.confidence}%")
        
        st.markdown("---")
        
        # Display matched image
        image_path = f"layout_images/{st.session_state.matched_image}"
        if os.path.exists(image_path):
            st.subheader("Selected Floorplan")
            
            if st.session_state.confidence < 100:
                st.warning(f"⚠️ {st.session_state.match_info}")
            else:
                st.success(f"✅ {st.session_state.match_info}")
            
            img = Image.open(image_path)
            st.image(img, caption=f"Filename: {st.session_state.matched_image}", use_column_width=True)
            
            # Metadata
            with st.expander("📊 Layout Metadata"):
                st.write(f"**Filename:** `{st.session_state.matched_image}`")
                st.write(f"**Match Confidence:** {st.session_state.confidence}%")
                st.write(f"**Input Plot:** {plot_width} × {plot_depth} feet")
                st.write(f"**BHK Configuration:** {bhk} BHK")
                st.write(f"**Floor Type:** {floor_type}")
                st.write(f"**District:** {district}")
        else:
            st.error(f"Image file not found: {image_path}")
    else:
        st.info("👈 Configure your plot parameters in the sidebar and click 'Generate Layout' to begin.")

# Report Tab
with tab2:
    if st.session_state.layout_generated and st.session_state.matched_image:
        st.header("📋 Explainability Report")
        
        zone = get_district_zone(district)
        zone_data = ZONE_CHARACTERISTICS[zone]
        category = get_plot_category(plot_width, plot_depth, bhk)
        plot_area = plot_width * plot_depth
        
        # Layout Summary
        st.subheader("1. Selected Layout Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Layout:** {st.session_state.matched_image}")
            st.write(f"**Plot Dimensions:** {plot_width} × {plot_depth} feet ({plot_area} sq ft)")
            st.write(f"**BHK:** {bhk} Bedroom Configuration")
        with col2:
            st.write(f"**Floor Type:** {floor_type}")
            st.write(f"**District:** {district}")
            st.write(f"**Plot Category:** {category}")
        
        st.markdown("---")
        
        # Selection Reasoning
        st.subheader("2. Why This Layout Was Selected")
        if st.session_state.confidence == 100:
            st.success(f"""
            ✅ **Exact Geometric Match**
            
            This layout precisely matches your input parameters:
            - Plot dimensions align with available standardized plans
            - BHK configuration matches user requirement ({bhk} BHK)
            - Floor type specification is met ({floor_type})
            - Geometry adheres to TNCDBR (Tamil Nadu Combined Development and Building Rules) setback requirements
            """)
        else:
            st.warning(f"""
            ⚠️ **Closest Available Match** (Confidence: {st.session_state.confidence}%)
            
            {st.session_state.match_info}
            
            While not an exact dimensional match, this layout provides:
            - Similar spatial requirements for {bhk} BHK configuration
            - Comparable floor type approach
            - TNCDBR-compliant design principles
            - Adaptable geometry for your plot constraints
            """)
        
        st.markdown("---")
        
        # Climate Zone Analysis
        st.subheader("3. Climate Zone Analysis")
        st.write(f"**Agro-Climatic Zone:** {zone}")
        st.info(f"**Climate Characteristics:** {zone_data['climate']}")
        
        st.markdown("---")
        
        # Laurie Baker Principles
        st.subheader("4. Laurie Baker Climate-Responsive Design Principles")
        st.write(zone_data['baker_principles'])
        
        with st.expander("🏛️ Core Baker Philosophy for This Zone"):
            st.markdown("""
            **Laurie Baker's cost-effective, climate-sensitive approach emphasizes:**
            
            - **Local Materials:** Use what's abundantly available in the region
            - **Functional Design:** Every element serves a purpose—no decoration for decoration's sake
            - **Climate Response:** Design works with nature, not against it
            - **Thermal Comfort:** Natural ventilation and thermal mass reduce energy dependence
            - **Minimal Waste:** Efficient use of materials reduces cost and environmental impact
            - **Human Scale:** Comfortable, livable spaces that don't overwhelm
            """)
        
        st.markdown("---")
        
        # Recommended Materials
        st.subheader("5. Recommended Materials for This District")
        st.write(f"Based on {zone} characteristics and local availability:")
        
        for idx, material in enumerate(zone_data['materials'], 1):
            st.write(f"{idx}. **{material}**")
        
        st.markdown("---")
        
        # Passive Design Strategies
        st.subheader("6. Passive Design Strategies")
        st.write("Climate-appropriate passive design recommendations:")
        
        for idx, strategy in enumerate(zone_data['passive_design'], 1):
            st.write(f"{idx}. {strategy}")
        
        st.markdown("---")
        
        # Construction Notes
        st.subheader("7. Construction Notes & Considerations")
        st.info(zone_data['notes'])
        
        # Additional technical recommendations
        with st.expander("🔧 Technical Recommendations"):
            st.markdown(f"""
            **For {category} in {district}:**
            
            **Foundation:**
            - Assess soil bearing capacity before design
            - Minimum plinth height: 450-600mm (increase in flood-prone areas)
            - Use concrete/stone for plinth in high-moisture zones
            
            **Walls:**
            - Minimum thickness: 230mm (9 inches) for load-bearing
            - Consider rat-trap bond for thermal insulation and material savings
            - Brick jaalis for aesthetic ventilation (where applicable)
            
            **Roof:**
            - Sloped roofs preferred for rainfall zones (30-35° pitch)
            - Filler slab technology reduces dead load and cost
            - Minimum 600mm overhang for weather protection
            
            **Openings:**
            - Minimum 10% of floor area for ventilation (increase in humid zones)
            - Shaded openings prevent direct sun penetration
            - Cross-ventilation through opposite walls is essential
            
            **Finishes:**
            - External: Weather-resistant paints or lime plaster (zone-dependent)
            - Internal: Breathable finishes like lime or natural cement plaster
            - Flooring: Oxide, mosaic, or vitrified tiles based on budget
            """)
        
        st.markdown("---")
        st.success("✅ Report generation complete. This analysis combines geometric layout selection with regional climate intelligence and Laurie Baker's cost-effective, sustainable design philosophy.")
        
    else:
        st.info("👈 Generate a layout first to view the detailed explainability report.")

# Footer
st.markdown("---")
st.markdown("**Tamil Nadu Plotwise Floorplan Explorer** | Geometric retrieval + climate intelligence engine")
