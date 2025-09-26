import os
import re
import shutil
import patoolib

# --- Configuration ---
# The folder where raw .rar demo files are located.
RAW_DEMO_DIR = "raw_demos"
# The main folder where sorted demos will be placed.
DEMO_BASE_DIR = "demos"
# A folder to move .rar files to after they've been processed.
PROCESSED_RAR_DIR = "processed_rar_files"
# A temporary folder for extractions. This will be created and deleted by the script.
TEMP_EXTRACT_DIR = "temp_extract"


def organize_demos(extraction_path, rar_filename_no_ext):
    """
    Scans the extracted folder for .dem files and moves them
    into the final organized folder structure.
    """
    print(f"    -> Organizing demo files from '{rar_filename_no_ext}'...")
    found_demos = False
    # Walk through all files and subdirectories in the extraction path
    for root, _, files in os.walk(extraction_path):
        for file in files:
            if file.endswith(".dem"):
                found_demos = True
                # Use regex to find the map name (e.g., de_inferno, de_nuke)
                match = re.search(r"(de_[a-zA-Z0-9_]+)", file)
                if match:
                    map_name = match.group(1).replace("de_", "")
                    # Create the final directory structure: demos/<map_name>/<original_rar_name>/
                    final_map_dir = os.path.join(DEMO_BASE_DIR, map_name)
                    final_demo_set_dir = os.path.join(
                        final_map_dir, rar_filename_no_ext
                    )
                    os.makedirs(final_demo_set_dir, exist_ok=True)

                    source_dem_path = os.path.join(root, file)
                    destination_dem_path = os.path.join(final_demo_set_dir, file)

                    print(f"      -> Moving '{file}' to '{final_demo_set_dir}'")
                    shutil.move(source_dem_path, destination_dem_path)
                else:
                    print(
                        f"      -> Could not determine map name for '{file}'. Skipping."
                    )

    if not found_demos:
        print("    -> No .dem files were found in this archive.")


def main():
    """Main function to find, extract, and sort demo archives."""
    print("--- HLTV Demo Sorter ---")

    # Create necessary directories if they don't exist
    os.makedirs(RAW_DEMO_DIR, exist_ok=True)
    os.makedirs(DEMO_BASE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_RAR_DIR, exist_ok=True)

    # Find all .rar files in the raw_demos directory
    try:
        rar_files_to_process = [
            f for f in os.listdir(RAW_DEMO_DIR) if f.lower().endswith(".rar")
        ]
    except FileNotFoundError:
        print(f"Error: The source folder '{RAW_DEMO_DIR}' was not found.")
        return

    if not rar_files_to_process:
        print(
            f"No .rar files found in the '{RAW_DEMO_DIR}' directory. Place your demo archives there and run again."
        )
        return

    print(
        f"Found {len(rar_files_to_process)} .rar file(s) in '{RAW_DEMO_DIR}' to process.\n"
    )

    for rar_file in rar_files_to_process:
        print(f"--- Processing '{rar_file}' ---")

        # Get the filename without the .rar extension
        rar_filename_no_ext = os.path.splitext(rar_file)[0]

        # Define a unique temporary path for this extraction
        current_extraction_path = os.path.join(TEMP_EXTRACT_DIR, rar_filename_no_ext)
        os.makedirs(current_extraction_path, exist_ok=True)

        source_rar_path = os.path.join(RAW_DEMO_DIR, rar_file)

        try:
            print(f"  -> Extracting '{rar_file}'...")
            patoolib.extract_archive(
                source_rar_path, outdir=current_extraction_path, verbosity=-1
            )
            print("  -> Extraction successful.")

            # Organize the extracted files
            organize_demos(current_extraction_path, rar_filename_no_ext)

            # Move the processed .rar file to the archive directory
            shutil.move(source_rar_path, os.path.join(PROCESSED_RAR_DIR, rar_file))
            print(f"  -> Moved '{rar_file}' to '{PROCESSED_RAR_DIR}' folder.\n")

        except patoolib.util.PatoolError as e:
            print(f"  -> ERROR: Failed to extract '{rar_file}'. Error: {e}")
            print(
                "  -> IMPORTANT: Ensure you have 'unrar' or '7zip'/'p7zip' installed and in your system's PATH."
            )
            print(f"  -> Skipping this file.\n")
        except Exception as e:
            print(f"  -> An unexpected error occurred: {e}\n")

    # Clean up the temporary extraction directory
    if os.path.exists(TEMP_EXTRACT_DIR):
        shutil.rmtree(TEMP_EXTRACT_DIR)

    print("--- Process Complete ---")


if __name__ == "__main__":
    main()
