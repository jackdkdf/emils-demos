"""Create map name to ID mapping."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

from .utils import normalize_map_name

logger = logging.getLogger(__name__)


def create_map_mapping(
    team_results_dir: Path,
    output_file: Path,
    num_files: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """Create map name to ID mapping from team results files.
    
    Args:
        team_results_dir: Directory containing team result CSV files.
        output_file: Path to save the mapping CSV.
        num_files: Number of files to process (default: 10).
        verbose: Whether to print progress messages.
        
    Returns:
        DataFrame with map_name and map_id columns.
    """
    # Get the first few CSV files from team_results
    team_results_files = sorted(team_results_dir.glob("*.csv"))
    num_files_to_process = min(num_files, len(team_results_files))
    files_to_process = team_results_files[:num_files_to_process]
    
    if verbose:
        logger.info(f"Processing {num_files_to_process} files to extract map information...")
        logger.info(f"Files: {[f.name for f in files_to_process]}\n")
    
    # Collect all map data from the files
    all_maps = []
    
    for file_path in files_to_process:
        try:
            df = pd.read_csv(file_path)
            if 'map_name' in df.columns and 'map_id' in df.columns:
                # Get unique map_name and map_id pairs
                maps = df[['map_name', 'map_id']].drop_duplicates()
                all_maps.append(maps)
                if verbose:
                    logger.info(f"  {file_path.name}: Found {len(maps)} unique maps")
        except Exception as e:
            if verbose:
                logger.warning(f"  Error reading {file_path.name}: {e}")
    
    # Combine all maps and get unique pairs
    if all_maps:
        combined_maps = pd.concat(all_maps, ignore_index=True)
        unique_maps = combined_maps.drop_duplicates(subset=['map_name', 'map_id'], keep='first')
        
        # Create mapping dataframe with normalized map names
        map_name_to_id = pd.DataFrame({
            'map_name': unique_maps['map_name'].apply(normalize_map_name),
            'map_id': unique_maps['map_id']
        })
        
        # Remove any duplicate map names (if any) - keep first occurrence
        map_name_to_id = map_name_to_id.drop_duplicates(subset='map_name', keep='first')
        
        # Sort by map_id for consistency
        map_name_to_id = map_name_to_id.sort_values('map_id').reset_index(drop=True)
        
        # Save to CSV
        map_name_to_id.to_csv(output_file, index=False)
        
        if verbose:
            logger.info(f"\nCreated map name to ID mapping with {len(map_name_to_id)} maps")
            logger.info(f"Saved to: {output_file}")
            logger.info(f"\nMap mappings:\n{map_name_to_id}")
        
        return map_name_to_id
    else:
        if verbose:
            logger.warning("No map data found in the processed files")
        return pd.DataFrame(columns=['map_name', 'map_id'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    team_results_dir = project_root / "data" / "raw" / "team_results"
    output_file = project_root / "data" / "mappings" / "map_name_to_id.csv"
    create_map_mapping(team_results_dir, output_file)
