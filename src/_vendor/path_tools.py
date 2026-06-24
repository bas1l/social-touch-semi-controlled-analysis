"""Vendored path utilities from the parent repo — only the functions needed by analysis."""

import os
import socket
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import json


def get_project_data_root():
    """
    Determines the project data root directory.

    First, it tries to find the directory automatically. If that fails or the
    directory doesn't exist, it opens a GUI dialog for the user to select it.
    The selected path is saved in a configuration file and used as the default
    path for the next time the function is called.

    Returns:
        Path: The path to the project_data_root directory.
        None: If the user cancels the directory selection.
    """
    config_file = "semi_controlled_project_data_config.json"

    # Load the saved path from the configuration file if it exists
    default_path = None
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            try:
                config = json.load(f)
                default_path = config.get("project_data_root")
            except json.JSONDecodeError:
                pass

    try:
        base_path = Path(_get_database_path())
        project_data_root = base_path / "semi-controlled"
        if project_data_root.is_dir():
            print(f"✅ Project DATA root automatically identified at: {project_data_root.resolve()}")
            return project_data_root
    except FileNotFoundError:
        pass

    print("⚠️ Project DATA root not found automatically.")
    print("Please select your 'semi-controlled' data folder using the dialog window.")

    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()

    selected_path = filedialog.askdirectory(
        title="Please Select the Project Data Folder",
        initialdir=default_path if default_path else "/"
    )

    root.destroy()

    if not selected_path:
        print("❌ No folder selected. Exiting program.")
        return None

    project_data_root = Path(selected_path)
    print(f"👍 Project DATA root set by user to: {project_data_root.resolve()}")

    with open(config_file, "w") as f:
        json.dump({"project_data_root": str(project_data_root)}, f)

    return project_data_root


def _get_onedrive_path_abs():
    if socket.gethostname() == "basil":
        onedrive_path_abs = os.path.join('F:\\', 'liu-onedrive-nospecial-carac', '_Teams')
    else:
        onedrive_path_abs = os.path.join('C:\\Users\\basdu83', 'OneDrive - Linköpings universitet', '_Teams')
    return onedrive_path_abs


def _get_team_path_abs(cloud_location="Teams"):
    onedrive_path_abs = _get_onedrive_path_abs()
    if cloud_location == "Teams":
        team_path = "Social touch Kinect MNG"
    elif cloud_location == "Sarah repository":
        team_path = os.path.join('touch comm MNG Kinect', 'basil_tmp')
    return os.path.join(onedrive_path_abs, team_path)


def _get_database_path(cloud_location="Teams"):
    team_path_abs = _get_team_path_abs(cloud_location=cloud_location)
    return os.path.join(team_path_abs, '02_data')
