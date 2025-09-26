import pandas as pd
from demoparser2 import DemoParser
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from urllib import request, error
from PIL import Image
import io
import os

# --- Calibrated Map Metadata ---
MAP_METADATA = {
    "de_inferno": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/1/11/CS2_inferno_radar.png/revision/latest?cb=20230901123108",
        "pos_x": -2037.19, "pos_y": 3835.81, "scale": 4.86,
    },
    "de_mirage": {
        "image_url": "https://raw.githubusercontent.com/LaihoE/cs-demo-min-viewer/main/public/images/maps/de_mirage.png",
        "pos_x": -3230, "pos_y": 1713, "scale": 5.0,
    },
    # Add other maps here if needed
}

# --- Utility and Output Functions (No changes here) ---
def get_map_image(map_name, map_info):
    image_dir = "map_images"
    image_path = os.path.join(image_dir, f"{map_name}.png")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(image_path):
        print(f"🖼️ Image for {map_name} not found locally. Downloading...")
        try:
            req = request.Request(map_info['image_url'], headers={'User-Agent': 'Mozilla/5.0'})
            with request.urlopen(req) as response, open(image_path, 'wb') as out_file:
                out_file.write(response.read())
            print("✅ Download complete.")
        except (error.HTTPError, Exception) as e:
            print(f"❌ Download failed: {e}.")
            return None
    return Image.open(image_path)

def print_grenade_log(df: pd.DataFrame):
    print("\n--- Grenade Landing Log ---")
    df_sorted = df.sort_values(by=["round_number", "tick"])
    current_round = -1
    for index, nade in df_sorted.iterrows():
        if "round_number" in nade and pd.notna(nade["round_number"]) and nade["round_number"] != current_round:
            current_round = nade["round_number"]
            print(f"\n{'='*15} Round {current_round} {'='*15}")
        user_name = nade.get("user_name", "Unknown")
        print(
            f"-> [{nade['time_formatted']}] {nade['grenade_type']:<12} by {user_name:<20} | "
            f"Landing Spot (X, Y, Z): ({nade['x']:.2f}, {nade['y']:.2f}, {nade['z']:.2f})"
        )

def plot_static_map(df: pd.DataFrame, map_name: str):
    if map_name not in MAP_METADATA: return
    map_info = MAP_METADATA[map_name]
    print(f"\n📈 Generating static plot for {map_name}...")
    df['pixel_x'] = (df['x'] - map_info['pos_x']) / map_info['scale']
    df['pixel_y'] = (map_info['pos_y'] - df['y']) / map_info['scale']
    map_image = get_map_image(map_name, map_info)
    if not map_image: return
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(map_image, extent=[0, 1024, 1024, 0])
    # Reverted palette to be specific
    palette = {"HE Grenade": "red", "Flashbang": "yellow", "Smoke": "limegreen", "Molotov": "orangered", "Incendiary": "orange", "Decoy": "grey"}
    sns.scatterplot(x='pixel_x', y='pixel_y', hue='grenade_type', data=df, ax=ax, palette=palette, s=80, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axis('off')
    plt.title(f"Grenade Landing Spots on {map_name.title()} (Static)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

def create_grenade_animation(df: pd.DataFrame, map_name: str):
    if map_name not in MAP_METADATA: return
    map_info = MAP_METADATA[map_name]
    print(f"\n🎥 Creating animation for {map_name}...")
    df_sorted = df.sort_values(by="tick")
    df_sorted['pixel_x'] = (df_sorted['x'] - map_info['pos_x']) / map_info['scale']
    df_sorted['pixel_y'] = (map_info['pos_y'] - df_sorted['y']) / map_info['scale']
    # Reverted palette to be specific
    palette = {"HE Grenade": "red", "Flashbang": "yellow", "Smoke": "limegreen", "Molotov": "orangered", "Incendiary": "orange", "Decoy": "grey"}
    df_sorted['color'] = df_sorted['grenade_type'].map(palette)
    map_image = get_map_image(map_name, map_info)
    if not map_image: return
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(map_image, extent=[0, 1024, 1024, 0])
    ax.axis('off')
    plt.title(f"Grenade Landing Spots on {map_name.title()} (Animated)", fontsize=16)
    scatter = ax.scatter([], [], s=80, alpha=0.8, edgecolor='black', linewidth=0.5)
    time_text = ax.text(5, 25, '', fontsize=14, color='white', ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
    def update(frame):
        current_data = df_sorted.iloc[:frame+1]
        scatter.set_offsets(current_data[['pixel_x', 'pixel_y']])
        scatter.set_color(current_data['color'])
        current_nade = df_sorted.iloc[frame]
        time_text.set_text(f"Round: {current_nade['round_number']}\nTime: {current_nade['time_formatted']}")
        print(f"Processing frame {frame + 1}/{len(df_sorted)}...", end='\r')
        return scatter, time_text
    anim = animation.FuncAnimation(fig, update, frames=len(df_sorted), interval=100, blit=True)
    try:
        output_filename = 'grenade_animation.mp4'
        anim.save(output_filename, writer='ffmpeg', fps=10, dpi=150)
        print(f"\n✅ Animation saved successfully as '{output_filename}'")
    except FileNotFoundError:
        print("\n❌ Error: ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH.")
    except Exception as e:
        print(f"\n❌ An error occurred while saving the animation: {e}")

# --- Main Analysis Function ---
def analyze_demo(demofile_path: str):
    """Parses a demo and generates a text log, a static plot, and an animation."""
    try:
        print(f"Parsing demo: {demofile_path}")
        parser = DemoParser(demofile_path)
        map_name = parser.parse_header()["map_name"]

        # --- REVERTED to specific event names ---
        event_names = [
            "round_start", "hegrenade_detonate", "flashbang_detonate",
            "smokegrenade_detonate", "decoy_started", "molotov_detonate", "incgrenade_detonate"
        ]
        event_dataframes = dict(parser.parse_events(event_names))
    except (FileNotFoundError, Exception) as e:
        print(f"❌ An error occurred during parsing: {e}")
        return

    # --- REVERTED to specific grenade mappings ---
    grenade_map = {
        "hegrenade_detonate": "HE Grenade",
        "flashbang_detonate": "Flashbang",
        "smokegrenade_detonate": "Smoke",
        "decoy_started": "Decoy",
        "molotov_detonate": "Molotov",
        "incgrenade_detonate": "Incendiary",
    }
    
    # Diagnostic summary to check event counts
    print("\n--- Found Grenade Events Summary ---")
    for event_name, friendly_name in grenade_map.items():
        df = event_dataframes.get(event_name)
        count = len(df) if df is not None else 0
        print(f"{friendly_name:<15}: {count} events found")

    all_grenades = []
    for event_name, friendly_name in grenade_map.items():
        df = event_dataframes.get(event_name)
        if df is not None and not df.empty:
            df["grenade_type"] = friendly_name
            all_grenades.append(df)

    if not all_grenades:
        print("\nNo grenade events to process.")
        return

    combined_df = pd.concat(all_grenades, ignore_index=True).dropna(subset=['x', 'y'])
    tickrate = parser.parse_header().get("tick_rate", 128)
    round_starts = event_dataframes.get("round_start", pd.DataFrame())
    if not round_starts.empty:
        round_start_ticks = round_starts["tick"].tolist()
        combined_df["round_number"] = pd.cut(combined_df["tick"], bins=round_start_ticks + [float("inf")], right=False, labels=range(1, len(round_start_ticks) + 1)).astype('Int64')
    
    def format_time(tick):
        total_seconds = tick / tickrate
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02}:{seconds:02}"
    combined_df["time_formatted"] = combined_df["tick"].apply(format_time)

    # Call all three output functions in sequence
    print_grenade_log(combined_df)
    plot_static_map(combined_df, map_name)
    create_grenade_animation(combined_df, map_name)

if __name__ == "__main__":
    DEMO_FILE = "demos/inferno/blast-austin-mouz-vs-vitality-m2-inferno.dem"
    analyze_demo(DEMO_FILE)