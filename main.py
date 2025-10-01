import pandas as pd
from demoparser2 import DemoParser
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
from urllib import request, error
from PIL import Image
import io
import os

# --- Calibrated Map Metadata (No changes here) ---
MAP_METADATA = {
    "de_ancient": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/9/9a/Ancient_Radar.png/revision/latest?cb=20221216234111",
        "pos_x": -2942.8,
        "pos_y": 2178.3,
        "scale": 5.002,
    },
    "de_dust2": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/0/0f/Cs2_dust2_overview.png/revision/latest?cb=20230323162128",
        "pos_x": -2479.0,
        "pos_y": 3242.9,
        "scale": 4.452,
    },
    "de_inferno": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/1/11/CS2_inferno_radar.png/revision/latest?cb=20230901123108",
        "pos_x": -2037.19,
        "pos_y": 3835.81,
        "scale": 4.86,
    },
    "de_mirage": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/9/95/Cs2_mirage_radar.png/revision/latest?cb=20231020111431",
        "pos_x": -3230,
        "pos_y": 1713,
        "scale": 5.0,
    },
    "de_nuke": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/8/8f/Cs2_nuke_radar.png/revision/latest?cb=20231020111944",
        "pos_x": -3554.0,
        "pos_y": 2971.5,
        "scale": 7.153,
    },
    "de_overpass": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/8/89/Cs2_overpass_radar.png/revision/latest?cb=20231020113838",
        "pos_x": -4832.6,
        "pos_y": 1781.2,
        "scale": 5.218,
    },
    "de_train": {
        "image_url": "https://static.wikia.nocookie.net/cswikia/images/8/8c/CS2_Train_radar.png/revision/latest?cb=20241114093521",
        "pos_x": -2297.4,
        "pos_y": 2072.0,
        "scale": 4.070,
    },
}


# --- Utility and Log Functions (No changes here) ---
def get_map_image(map_name, map_info):
    image_dir = "map_images"
    image_path = os.path.join(image_dir, f"{map_name}.png")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(image_path):
        print(f"🖼️ Image for {map_name} not found locally. Downloading...")
        try:
            req = request.Request(
                map_info["image_url"], headers={"User-Agent": "Mozilla/5.0"}
            )
            with request.urlopen(req) as response, open(image_path, "wb") as out_file:
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
        if (
            "round_number" in nade
            and pd.notna(nade["round_number"])
            and nade["round_number"] != current_round
        ):
            current_round = nade["round_number"]
            print(f"\n{'='*15} Round {current_round} {'='*15}")
        user_name = nade.get("user_name", "Unknown")
        print(
            f"-> [{nade['time_formatted']}] {nade['grenade_type']:<12} by {user_name:<20} | "
            f"Landing Spot (X, Y, Z): ({nade['x']:.2f}, {nade['y']:.2f}, {nade['z']:.2f})"
        )


# --- INTERACTIVE PLOTTER CLASS (No changes here) ---
class InteractivePlotter:
    def __init__(self, df: pd.DataFrame, map_name: str):
        self.df = df
        self.map_name = map_name
        self.map_info = MAP_METADATA[map_name]
        self.palette = {
            "HE Grenade": "red",
            "Flashbang": "yellow",
            "Smoke": "limegreen",
            "Molotov": "orangered",
            "Incendiary": "orange",
            "Decoy": "grey",
        }

        self.df["pixel_x"] = (self.df["x"] - self.map_info["pos_x"]) / self.map_info[
            "scale"
        ]
        self.df["pixel_y"] = (self.map_info["pos_y"] - self.df["y"]) / self.map_info[
            "scale"
        ]
        self.df["color"] = self.df["grenade_type"].map(self.palette)

        # Filter out rounds with no data before creating the list
        self.rounds = sorted(self.df["round_number"].dropna().unique())
        if not self.rounds:
            print("No valid rounds with grenade data to display.")
            return
        self.current_round_index = 0

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        plt.subplots_adjust(bottom=0.15)

        map_image = get_map_image(self.map_name, self.map_info)
        if map_image:
            self.ax.imshow(map_image, extent=[0, 1024, 1024, 0])

        self.scatter = self.ax.scatter(
            [], [], s=100, alpha=0.9, edgecolor="black", linewidth=0.5
        )
        self.annotations = []

        self.ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, "Previous")
        self.btn_next = Button(self.ax_next, "Next")
        self.btn_prev.on_clicked(self.prev_round)
        self.btn_next.on_clicked(self.next_round)

        self.update_plot()

    def update_plot(self):
        if not self.rounds:
            return
        for ann in self.annotations:
            ann.remove()
        self.annotations = []

        current_round = self.rounds[self.current_round_index]
        round_df = self.df[self.df["round_number"] == current_round]

        self.scatter.set_offsets(round_df[["pixel_x", "pixel_y"]])
        self.scatter.set_color(round_df["color"])

        for index, row in round_df.iterrows():
            label = (
                f"{row.get('user_team_abbr', '?')}\n{row.get('time_in_round', '?:??')}"
            )
            ann = self.ax.text(
                row["pixel_x"] + 8,
                row["pixel_y"],
                label,
                fontsize=8,
                color="white",
                fontweight="bold",
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="none"),
            )
            self.annotations.append(ann)

        self.ax.set_title(
            f"Grenades in Round {current_round} on {self.map_name}", fontsize=16
        )
        self.fig.canvas.draw_idle()

    def next_round(self, event):
        if not self.rounds:
            return
        self.current_round_index = (self.current_round_index + 1) % len(self.rounds)
        self.update_plot()

    def prev_round(self, event):
        if not self.rounds:
            return
        self.current_round_index = (
            self.current_round_index - 1 + len(self.rounds)
        ) % len(self.rounds)
        self.update_plot()


# --- MAIN ANALYSIS FUNCTION ---
def analyze_demo(demofile_path: str):
    try:
        print(f"Parsing demo: {demofile_path}")
        parser = DemoParser(demofile_path)
        map_name = parser.parse_header()["map_name"]
        event_names = [
            "round_start",
            "player_team",
            "hegrenade_detonate",
            "flashbang_detonate",
            "smokegrenade_detonate",
            "decoy_started",
            "molotov_detonate",
            "incgrenade_detonate",
        ]
        event_dataframes = dict(parser.parse_events(event_names))
    except (FileNotFoundError, Exception) as e:
        print(f"❌ An error occurred during parsing: {e}")
        return None, None

    grenade_map = {
        "hegrenade_detonate": "HE Grenade",
        "flashbang_detonate": "Flashbang",
        "smokegrenade_detonate": "Smoke",
        "decoy_started": "Decoy",
        "molotov_detonate": "Molotov",
        "incgrenade_detonate": "Incendiary",
    }

    all_grenades = []
    for event_name, friendly_name in grenade_map.items():
        df = event_dataframes.get(event_name)
        if df is not None and not df.empty:
            df["grenade_type"] = friendly_name
            all_grenades.append(df)

    if not all_grenades:
        print("\nNo grenade events to process.")
        return None, None

    combined_df = pd.concat(all_grenades, ignore_index=True).dropna(subset=["x", "y"])
    tickrate = parser.parse_header().get("tick_rate", 128)
    round_starts = event_dataframes.get("round_start", pd.DataFrame())

    if not round_starts.empty:
        round_starts = round_starts.sort_values(by="tick").reset_index(drop=True)
        round_starts["round_number"] = round_starts.index + 1

        round_start_ticks = round_starts["tick"].tolist()
        bins = round_start_ticks + [float("inf")]
        labels = round_starts["round_number"].tolist()
        combined_df["round_number"] = pd.cut(
            combined_df["tick"], bins=bins, right=False, labels=labels
        )

        combined_df.dropna(subset=["round_number"], inplace=True)
        combined_df["round_number"] = combined_df["round_number"].astype("Int64")

        round_start_map = round_starts.set_index("round_number")["tick"]
        combined_df["round_start_tick"] = combined_df["round_number"].map(
            round_start_map
        )

        tick_in_round = combined_df["tick"] - combined_df["round_start_tick"]
        seconds_in_round = tick_in_round / tickrate

        def format_seconds(s):
            if pd.isna(s) or s < 0:
                return "0:00"
            minutes = int(s // 60)
            seconds = int(s % 60)
            return f"{minutes}:{seconds:02d}"

        combined_df["time_in_round"] = seconds_in_round.apply(format_seconds)

        # --- MODIFIED: Added data type enforcement and debugging ---
        player_team_df = event_dataframes.get("player_team")

        # Check for necessary columns in both dataframes
        if (
            player_team_df is not None
            and not player_team_df.empty
            and "team" in player_team_df.columns
            and "user_steamid" in player_team_df.columns
            and "user_steamid" in combined_df.columns
        ):

            # --- FIX: Enforce consistent data types for the join key ---
            player_team_df.dropna(subset=["user_steamid"], inplace=True)
            combined_df.dropna(subset=["user_steamid"], inplace=True)
            player_team_df["user_steamid"] = player_team_df["user_steamid"].astype(
                "Int64"
            )
            combined_df["user_steamid"] = combined_df["user_steamid"].astype("Int64")

            player_team_df["round_number"] = pd.cut(
                player_team_df["tick"], bins=bins, right=False, labels=labels
            )
            player_team_df.dropna(subset=["round_number"], inplace=True)
            player_team_df["round_number"] = player_team_df["round_number"].astype(
                "Int64"
            )

            team_lookup = player_team_df.drop_duplicates(
                subset=["round_number", "user_steamid"], keep="last"
            ).set_index(["round_number", "user_steamid"])["team"]

            grenade_index = pd.MultiIndex.from_frame(
                combined_df[["round_number", "user_steamid"]]
            )
            combined_df["team_num"] = grenade_index.map(team_lookup)

            team_num_map = {2: "T", 3: "CT"}
            combined_df["user_team_abbr"] = (
                combined_df["team_num"].map(team_num_map).fillna("?")
            )

            # --- AGGRESSIVE DEBUGGING (In case it still fails) ---
            print("\n\n" + "=" * 20 + " DEBUGGING OUTPUT " + "=" * 20)
            print(
                f"\nSuccessfully found and used 'team' column. Lookups performed: {len(combined_df)}"
            )
            print(
                "Number of successful team mappings:",
                (combined_df["user_team_abbr"] != "?").sum(),
            )
            print("Sample of final data:")
            print(
                combined_df[
                    [
                        "round_number",
                        "user_name",
                        "user_steamid",
                        "team_num",
                        "user_team_abbr",
                    ]
                ].head()
            )
            print("\n" + "=" * 20 + " END DEBUGGING " + "=" * 25 + "\n")

        else:
            combined_df["user_team_abbr"] = "?"
            print(
                "Could not perform team lookup: 'player_team' data or 'user_steamid' column was missing."
            )

    def format_time(tick):
        total_seconds = tick / tickrate
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02}:{seconds:02}"

    combined_df["time_formatted"] = combined_df["tick"].apply(format_time)

    return combined_df, map_name


if __name__ == "__main__":
    DEMO_FILE = r"demos\train\fissure-playground-2-virtuspro-vs-astralis-bo3-C6yzLSUiwN_qLnOK4Pb373\virtuspro-vs-astralis-m2-train.dem"

    full_grenade_df, map_name = analyze_demo(DEMO_FILE)

    if full_grenade_df is not None:
        print_grenade_log(full_grenade_df)
        print("\n launching interactive plot...")
        # Check if there's any data left to plot
        if not full_grenade_df.empty:
            plotter = InteractivePlotter(full_grenade_df, map_name)
            plt.show()
        else:
            print("No grenade data available to plot after processing.")
