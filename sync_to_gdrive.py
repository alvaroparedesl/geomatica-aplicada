#!/usr/bin/env python3
"""
Sync Jupyter Notebooks to Google Drive for Google Colab

This script converts .py files (Jupytext format) to .ipynb and uploads them to Google Drive.
It can handle individual files, folders, or the entire repository.

Usage:
    python sync_to_gdrive.py setup                    # Setup Google Drive credentials
    python sync_to_gdrive.py sync --all              # Sync all notebooks
    python sync_to_gdrive.py sync --folder path      # Sync specific folder
    python sync_to_gdrive.py sync --file path        # Sync specific file
    python sync_to_gdrive.py sync --all --dry-run    # Show what would be synced
"""

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jupytext
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages:")
    print(
        "pip install jupytext google-api-python-client google-auth-httplib2 google-auth-oauthlib"
    )
    sys.exit(1)

# Configuration
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.pickle"
CONFIG_FILE = "gdrive_config.json"
DEFAULT_FOLDER_NAME = "Geomatica-Aplicada-Notebooks"


class GoogleDriveSync:
    def __init__(self):
        self.service = None
        self.gdrive_folder_id = None
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config: {e}")

    def extract_folder_id_from_url(self, url: str) -> Optional[str]:
        """Extract folder ID from Google Drive URL"""
        # Pattern for Google Drive folder URLs
        patterns = [
            r"drive\.google\.com/drive/folders/([a-zA-Z0-9-_]+)",
            r"drive\.google\.com/drive/u/\d+/folders/([a-zA-Z0-9-_]+)",
            r"drive\.google\.com/open\?id=([a-zA-Z0-9-_]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def validate_folder_id(self, folder_id: str) -> bool:
        """Validate that a folder ID exists and is accessible"""
        try:
            # Ensure we have a service connection
            if not self.service:
                if not self.authenticate():
                    print("✗ Failed to authenticate with Google Drive")
                    return False

            # Validate folder ID format (basic check)
            if not folder_id or len(folder_id) < 10:
                print("✗ Invalid folder ID format")
                return False

            print(f"🔍 Checking folder access for ID: {folder_id}")

            # Try to get folder metadata (including shared drives)
            folder = (
                self.service.files()
                .get(
                    fileId=folder_id,
                    fields="id,name,mimeType,parents",
                    supportsAllDrives=True,
                )
                .execute()
            )
            mime_type = folder.get("mimeType")
            folder_name = folder.get("name", "Unknown")

            if mime_type == "application/vnd.google-apps.folder":
                print(f"✓ Folder validated: '{folder_name}' (ID: {folder_id})")
                return True
            else:
                print(
                    f"✗ ID points to a file, not a folder: '{folder_name}' (type: {mime_type})"
                )
                return False

        except Exception as e:
            error_msg = str(e)

            # Parse different types of errors
            if "File not found" in error_msg or "notFound" in error_msg:
                print(f"✗ Folder not found or no access: {folder_id}")
                print("  Possible causes:")
                print("  - Folder ID is incorrect")
                print("  - Folder was deleted")
                print("  - You don't have permission to access this folder")
                print("  - Folder is in a different Google account")
                print(
                    "  - Folder is in a shared drive (Team Drive) with restricted access"
                )
                print("\n  💡 Troubleshooting tips:")
                print("  - Make sure you're logged into the correct Google account")
                print("  - Try creating a test folder and use its ID")
                print("  - Check if the folder URL is complete and correct")
                print("  - Verify the folder is shared with your account")
                print(
                    "  - If folder is in a shared drive, ensure you have proper access"
                )
            elif "forbidden" in error_msg.lower():
                print(f"✗ Access denied to folder: {folder_id}")
                print("  You don't have permission to access this folder")
                print("  - Request access from the folder owner")
                print("  - Make sure you're authenticated with the correct account")
            elif "invalid" in error_msg.lower():
                print(f"✗ Invalid folder ID format: {folder_id}")
            else:
                print(f"✗ Folder validation failed: {error_msg}")

            return False

    def test_folder_access(self, folder_id: str) -> Dict[str, Any]:
        """Test folder access and return detailed information"""
        result = {
            "accessible": False,
            "exists": False,
            "is_folder": False,
            "name": None,
            "error": None,
            "suggestions": [],
        }

        try:
            if not self.service:
                if not self.authenticate():
                    result["error"] = "Authentication failed"
                    return result

            # Try to get folder metadata (including shared drives)
            folder = (
                self.service.files()
                .get(
                    fileId=folder_id,
                    fields="id,name,mimeType,parents,capabilities",
                    supportsAllDrives=True,
                )
                .execute()
            )

            result["exists"] = True
            result["name"] = folder.get("name", "Unknown")
            mime_type = folder.get("mimeType")
            result["is_folder"] = mime_type == "application/vnd.google-apps.folder"

            if result["is_folder"]:
                result["accessible"] = True
            else:
                result["error"] = (
                    f"ID points to a file, not a folder (type: {mime_type})"
                )
                result["suggestions"].append(
                    "Make sure you copied the folder ID, not a file ID"
                )

        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg

            if "File not found" in error_msg or "notFound" in error_msg:
                result["suggestions"].extend(
                    [
                        "Double-check the folder ID is correct",
                        "Make sure the folder exists and wasn't deleted",
                        "Verify you have access to the folder",
                        "Check if the folder is in the same Google account you're authenticated with",
                        "If folder is in a shared drive, ensure you have proper access permissions",
                    ]
                )
            elif "forbidden" in error_msg.lower():
                result["suggestions"].extend(
                    [
                        "Request access to the folder from the owner",
                        "Make sure you're authenticated with the correct Google account",
                        "Check if the folder sharing settings allow your access",
                    ]
                )

        return result

    def create_test_folder(self) -> Optional[str]:
        """Create a test folder for users to verify access"""
        try:
            if not self.service:
                if not self.authenticate():
                    print("✗ Failed to authenticate")
                    return None

            test_folder_name = f"Test-Folder-{DEFAULT_FOLDER_NAME}"

            # Create test folder
            file_metadata = {
                "name": test_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }

            folder = (
                self.service.files()
                .create(body=file_metadata, fields="id,name")
                .execute()
            )
            folder_id = folder.get("id")
            folder_name = folder.get("name")

            print(f"✓ Created test folder: '{folder_name}'")
            print(f"  Folder ID: {folder_id}")
            print(f"  URL: https://drive.google.com/drive/folders/{folder_id}")
            print("\n  You can now use this folder ID for configuration.")
            print("  You can rename or delete this folder later if needed.")

            return folder_id

        except Exception as e:
            print(f"✗ Failed to create test folder: {e}")
            return None

    def list_shared_drives(self):
        """List available shared drives for debugging"""
        try:
            if not self.service:
                if not self.authenticate():
                    return []

            # List shared drives
            results = self.service.drives().list().execute()
            drives = results.get("drives", [])

            print(f"Found {len(drives)} shared drives:")
            for drive in drives:
                print(f"  - {drive['name']} (ID: {drive['id']})")

            return drives

        except Exception as e:
            print(f"Error listing shared drives: {e}")
            return []

    def setup_credentials(self, force_refresh: bool = False):
        """Setup Google Drive API credentials and folder configuration"""
        print("Setting up Google Drive credentials and folder configuration...")

        # Check if we already have valid credentials
        if not force_refresh and os.path.exists(TOKEN_FILE):
            print("✓ Found existing authentication token")
            if self.authenticate():
                print("✓ Existing credentials are valid")
                self.setup_folder_configuration()
                return True
            else:
                print("⚠ Existing credentials are invalid, setting up new ones...")

        print("\nStep 1: Enable Google Drive API")
        print("1. Go to https://console.developers.google.com/")
        print("2. Create a new project or select an existing one")
        print("3. Enable the Google Drive API")
        print("4. Go to 'Credentials' > 'Create Credentials' > 'OAuth 2.0 Client IDs'")
        print("5. Choose 'Desktop application'")
        print(
            "6. Download the JSON file and save it as 'credentials.json' in this directory"
        )

        input("\nPress Enter when you have saved the credentials.json file...")

        if not os.path.exists(CREDENTIALS_FILE):
            print(f"Error: {CREDENTIALS_FILE} not found!")
            print(
                "Please download the credentials file and save it as 'credentials.json'"
            )
            return False

        try:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)

            print("✓ Credentials setup successfully!")

            # Setup folder configuration
            self.service = build("drive", "v3", credentials=creds)
            self.setup_folder_configuration()

            return True

        except Exception as e:
            print(f"Error setting up credentials: {e}")
            return False

    def setup_folder_configuration(self):
        """Configure the target Google Drive folder"""
        print("\nStep 2: Configure Google Drive folder")
        print("You can either:")
        print("1. Use an existing folder (provide folder ID or URL)")
        print("2. Create a test folder to verify access")
        print("3. Create a new folder with default name")

        choice = input("\nEnter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            print("\nTo get a folder ID:")
            print("1. Open Google Drive in your browser")
            print("2. Navigate to the folder you want to use")
            print("3. Copy the URL from the address bar")
            print(
                "   Example: https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74mMjoeqXVE"
            )
            print("4. Or just copy the folder ID (the long string after 'folders/')")

            folder_input = input("\nEnter folder URL or folder ID: ").strip()

            # Try to extract folder ID from URL
            folder_id = self.extract_folder_id_from_url(folder_input)
            if not folder_id:
                # Assume it's already a folder ID
                folder_id = folder_input

            # Test folder access with detailed feedback
            access_result = self.test_folder_access(folder_id)

            if access_result["accessible"]:
                folder_name = access_result["name"]
                self.config["folder_id"] = folder_id
                self.config["folder_name"] = folder_name
                self.save_config(self.config)

                print(f"✓ Configured to use folder: '{folder_name}' (ID: {folder_id})")
                return True
            else:
                print(f"✗ Cannot access folder: {folder_id}")
                if access_result["error"]:
                    print(f"  Error: {access_result['error']}")

                if access_result["suggestions"]:
                    print("  Suggestions:")
                    for suggestion in access_result["suggestions"]:
                        print(f"  - {suggestion}")

                        # Offer to try again or create new folder
                print("\nOptions:")
                print("1. Try a different folder ID")
                print("2. Create a test folder to verify access")
                print("3. Create a new folder with default name")
                choice = input("Enter your choice (1, 2, or 3): ").strip()

                if choice == "1":
                    return (
                        self.setup_folder_configuration()
                    )  # Recursive call to try again
                elif choice == "2":
                    # Create test folder
                    test_folder_id = self.create_test_folder()
                    if test_folder_id:
                        use_test = (
                            input("\nUse this test folder for sync? (y/N): ")
                            .strip()
                            .lower()
                        )
                        if use_test in ["y", "yes"]:
                            # Use the test folder
                            access_result = self.test_folder_access(test_folder_id)
                            if access_result["accessible"]:
                                self.config["folder_id"] = test_folder_id
                                self.config["folder_name"] = access_result["name"]
                                self.save_config(self.config)
                                print(
                                    f"✓ Configured to use test folder: '{access_result['name']}'"
                                )
                                return True
                        else:
                            print("Test folder created but not configured for sync.")
                            return self.setup_folder_configuration()
                    return False
                elif choice == "3":
                    # Fall through to create new folder option
                    pass
                else:
                    print("Invalid choice. Setup cancelled.")
                    return False

        elif choice == "2":
            # Create test folder
            test_folder_id = self.create_test_folder()
            if test_folder_id:
                use_test = (
                    input("\nUse this test folder for sync? (y/N): ").strip().lower()
                )
                if use_test in ["y", "yes"]:
                    # Use the test folder
                    access_result = self.test_folder_access(test_folder_id)
                    if access_result["accessible"]:
                        self.config["folder_id"] = test_folder_id
                        self.config["folder_name"] = access_result["name"]
                        self.save_config(self.config)
                        print(
                            f"✓ Configured to use test folder: '{access_result['name']}'"
                        )
                        return True
                else:
                    print("Test folder created but not configured for sync.")
                    return self.setup_folder_configuration()
            return False

        elif choice == "3":
            folder_name = input(
                f"\nEnter folder name (default: {DEFAULT_FOLDER_NAME}): "
            ).strip()
            if not folder_name:
                folder_name = DEFAULT_FOLDER_NAME

            self.config["folder_name"] = folder_name
            # Don't set folder_id yet - it will be created during first sync
            if "folder_id" in self.config:
                del self.config["folder_id"]
            self.save_config(self.config)

            print(f"✓ Configured to create/use folder: {folder_name}")
            return True

        else:
            print("Invalid choice. Please run setup again.")
            return False

    def show_status(self):
        """Show current configuration status"""
        print("Current Configuration:")
        print("=" * 50)

        # Check credentials
        if os.path.exists(CREDENTIALS_FILE):
            print("✓ Credentials file: Found")
        else:
            print("✗ Credentials file: Not found")

        if os.path.exists(TOKEN_FILE):
            print("✓ Authentication token: Found")
        else:
            print("✗ Authentication token: Not found")

        # Check configuration
        if self.config:
            folder_id = self.config.get("folder_id")
            folder_name = self.config.get("folder_name", "Not configured")

            print(f"📁 Target folder name: {folder_name}")
            if folder_id:
                print(f"🆔 Target folder ID: {folder_id}")

                # Validate folder if we can authenticate
                if self.authenticate():
                    if self.validate_folder_id(folder_id):
                        print("✓ Folder access: Valid")
                    else:
                        print("✗ Folder access: Invalid or no permission")
            else:
                print("🆔 Target folder ID: Will be created on first sync")
        else:
            print("📁 Target folder: Not configured")

        print("=" * 50)

        if not os.path.exists(CREDENTIALS_FILE) or not self.config:
            print("\nRun 'setup' to configure credentials and folder")
        elif not self.config.get("folder_id") and not self.config.get("folder_name"):
            print("\nRun 'config' to configure the target folder")

    def reset_credentials(self):
        """Reset/remove stored credentials"""
        files_removed = []

        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
            files_removed.append("authentication token")

        if os.path.exists(CREDENTIALS_FILE):
            choice = (
                input("Do you want to remove credentials.json as well? (y/N): ")
                .strip()
                .lower()
            )
            if choice in ["y", "yes"]:
                os.remove(CREDENTIALS_FILE)
                files_removed.append("credentials file")

        if files_removed:
            print(f"✓ Removed: {', '.join(files_removed)}")
        else:
            print("No credential files found to remove")

    def authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None

        # Load existing token
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)

        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(CREDENTIALS_FILE):
                    print("Credentials not found. Please run 'setup' first.")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES
                )
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)

        self.service = build("drive", "v3", credentials=creds)
        return True

    def find_or_create_folder(self, folder_name: str, parent_id: str = None) -> str:
        """Find or create a folder in Google Drive"""
        # Search for existing folder
        query = (
            f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        else:
            query += " and 'root' in parents"

        results = (
            self.service.files()
            .list(q=query, supportsAllDrives=True, includeItemsFromAllDrives=True)
            .execute()
        )
        items = results.get("files", [])

        if items:
            return items[0]["id"]

        # Create folder if it doesn't exist
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            file_metadata["parents"] = [parent_id]

        folder = (
            self.service.files()
            .create(body=file_metadata, fields="id", supportsAllDrives=True)
            .execute()
        )
        return folder.get("id")

    def setup_folder_structure(self):
        """Create the main folder and get its ID"""
        # Check if we have a configured folder ID
        folder_id = self.config.get("folder_id")
        folder_name = self.config.get("folder_name", DEFAULT_FOLDER_NAME)

        if folder_id:
            # Validate the configured folder ID
            if self.validate_folder_id(folder_id):
                self.gdrive_folder_id = folder_id
                print(f"✓ Using configured folder (ID: {self.gdrive_folder_id})")
                return self.gdrive_folder_id
            else:
                print(
                    f"⚠ Configured folder ID {folder_id} is not valid, creating new folder..."
                )

        # Create or find folder by name
        self.gdrive_folder_id = self.find_or_create_folder(folder_name)
        print(f"✓ Main folder '{folder_name}' ready (ID: {self.gdrive_folder_id})")

        # Save the folder ID for future use
        self.config["folder_id"] = self.gdrive_folder_id
        self.config["folder_name"] = folder_name
        self.save_config(self.config)

        return self.gdrive_folder_id

    def upload_file(
        self, local_path: str, gdrive_folder_id: str, filename: str = None
    ) -> bool:
        """Upload a file to Google Drive"""
        if filename is None:
            filename = os.path.basename(local_path)

        try:
            # First, check if file already exists in the target folder
            query = f"name='{filename}' and '{gdrive_folder_id}' in parents"
            results = (
                self.service.files()
                .list(
                    q=query,
                    fields="files(id,name,parents)",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items = results.get("files", [])

            # If not found in target folder, search by name only
            if not items:
                query_by_name = f"name='{filename}'"
                results_by_name = (
                    self.service.files()
                    .list(
                        q=query_by_name,
                        fields="files(id,name,parents)",
                        supportsAllDrives=True,
                        includeItemsFromAllDrives=True,
                    )
                    .execute()
                )
                items_by_name = results_by_name.get("files", [])

                if items_by_name:
                    # File exists elsewhere, we'll update and move it
                    items = items_by_name
                    print(
                        f"📁 Found {filename} in different folder, will move and update"
                    )

            media = MediaFileUpload(local_path, resumable=True)

            if items:
                # Update existing file
                file_id = items[0]["id"]
                existing_parents = items[0].get("parents", [])

                # For updates, don't include parents in body
                file_metadata = {"name": filename}

                # Check if we need to move the file to a different folder
                if gdrive_folder_id not in existing_parents:
                    # File needs to be moved to the target folder
                    file = (
                        self.service.files()
                        .update(
                            fileId=file_id,
                            body=file_metadata,
                            media_body=media,
                            addParents=gdrive_folder_id,
                            removeParents=",".join(existing_parents),
                            supportsAllDrives=True,
                        )
                        .execute()
                    )
                else:
                    # File is already in the correct folder, just update content
                    file = (
                        self.service.files()
                        .update(
                            fileId=file_id,
                            body=file_metadata,
                            media_body=media,
                            supportsAllDrives=True,
                        )
                        .execute()
                    )
                print(f"✓ Updated: {filename}")
            else:
                # Create new file - parents field is allowed here
                file_metadata = {"name": filename, "parents": [gdrive_folder_id]}
                file = (
                    self.service.files()
                    .create(
                        body=file_metadata,
                        media_body=media,
                        fields="id",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
                print(f"✓ Uploaded: {filename}")

            return True

        except Exception as e:
            print(f"✗ Error uploading {filename}: {e}")
            return False

    def convert_py_to_ipynb(self, py_file: str, force: bool = False) -> Optional[str]:
        """Convert .py file to .ipynb using jupytext"""
        py_path = Path(py_file)
        ipynb_path = py_path.with_suffix(".ipynb")

        # Check if conversion is needed
        if ipynb_path.exists() and not force:
            if ipynb_path.stat().st_mtime > py_path.stat().st_mtime:
                print(f"⚡ Skipping {py_file} (up to date)")
                return str(ipynb_path)

        try:
            # Read and convert
            notebook = jupytext.read(py_file)
            jupytext.write(notebook, ipynb_path)
            print(f"🔄 Converted: {py_file} → {ipynb_path.name}")
            return str(ipynb_path)

        except Exception as e:
            print(f"✗ Error converting {py_file}: {e}")
            return None

    def find_py_files(self, path: str) -> List[str]:
        """Find all .py files in the given path"""
        path_obj = Path(path)

        if path_obj.is_file():
            if path_obj.suffix == ".py":
                return [str(path_obj)]
            else:
                return []

        py_files = []
        for py_file in path_obj.rglob("*.py"):
            # Skip __pycache__ and other system files
            if "__pycache__" not in str(py_file) and not py_file.name.startswith("."):
                py_files.append(str(py_file))

        return sorted(py_files)

    def create_gdrive_folder_structure(
        self, local_path: str, base_folder_id: str
    ) -> str:
        """Create folder structure in Google Drive matching local structure"""
        local_path = Path(local_path)
        current_folder_id = base_folder_id

        # Get relative path from notebooks directory
        try:
            rel_path = local_path.relative_to("notebooks")
            parts = rel_path.parts[:-1]  # Exclude filename

            for part in parts:
                current_folder_id = self.find_or_create_folder(part, current_folder_id)

        except ValueError:
            # File is not in notebooks directory, create structure as-is
            parts = local_path.parts[:-1]
            for part in parts:
                current_folder_id = self.find_or_create_folder(part, current_folder_id)

        return current_folder_id

    def sync_files(self, paths: List[str], dry_run: bool = False, force: bool = False):
        """Sync files to Google Drive"""

        # For dry run, we don't need authentication
        if not dry_run:
            if not self.authenticate():
                return False

            if not self.setup_folder_structure():
                return False

        total_files = 0
        converted_files = 0
        uploaded_files = 0

        for path in paths:
            py_files = self.find_py_files(path)
            total_files += len(py_files)

            for py_file in py_files:
                print(f"\nProcessing: {py_file}")

                if dry_run:
                    print(
                        f"  Would convert: {py_file} → {Path(py_file).with_suffix('.ipynb').name}"
                    )
                    # Check if folder structure would be created
                    try:
                        rel_path = Path(py_file).relative_to("notebooks")
                        if len(rel_path.parts) > 1:
                            folder_path = "/".join(rel_path.parts[:-1])
                            print(f"  Would create folder: {folder_path}")
                    except ValueError:
                        print("  Would upload to: root folder")
                    converted_files += 1
                    continue

                # Convert to .ipynb
                ipynb_file = self.convert_py_to_ipynb(py_file, force)
                if not ipynb_file:
                    continue

                converted_files += 1

                # Create folder structure in Google Drive
                folder_id = self.create_gdrive_folder_structure(
                    py_file, self.gdrive_folder_id
                )

                # Upload to Google Drive
                if self.upload_file(ipynb_file, folder_id):
                    uploaded_files += 1

        print(f"\n{'=' * 50}")
        print("Summary:")
        print(f"Total .py files found: {total_files}")
        print(f"Files {'would be ' if dry_run else ''}converted: {converted_files}")
        if not dry_run:
            print(f"Files uploaded: {uploaded_files}")
        else:
            print("(Dry run - no files were actually uploaded)")
        print(f"{'=' * 50}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync Jupyter Notebooks to Google Drive"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Setup Google Drive credentials and folder"
    )
    setup_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh of credentials, ignore existing token",
    )

    # Configure command
    config_parser = subparsers.add_parser(
        "config", help="Configure Google Drive folder"
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current configuration")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset stored credentials")

    # Debug command
    debug_parser = subparsers.add_parser("debug", help="Debug shared drive access")

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Test file upload/update functionality"
    )
    test_parser.add_argument(
        "--file", type=str, required=True, help="Test file to upload"
    )

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync notebooks to Google Drive")
    sync_group = sync_parser.add_mutually_exclusive_group(required=True)
    sync_group.add_argument("--all", action="store_true", help="Sync all notebooks")
    sync_group.add_argument("--folder", type=str, help="Sync specific folder")
    sync_group.add_argument("--file", type=str, help="Sync specific file")
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without doing it",
    )
    sync_parser.add_argument(
        "--force", action="store_true", help="Force reconversion of .ipynb files"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    syncer = GoogleDriveSync()

    if args.command == "setup":
        force_refresh = getattr(args, "force_refresh", False)
        syncer.setup_credentials(force_refresh)

    elif args.command == "config":
        if not syncer.authenticate():
            print("Please run 'setup' first to configure credentials")
            return
        syncer.setup_folder_configuration()

    elif args.command == "status":
        syncer.show_status()

    elif args.command == "reset":
        syncer.reset_credentials()

    elif args.command == "debug":
        print("Debugging shared drive access...")
        syncer.list_shared_drives()

        # Test the problematic folder
        folder_id = "1RzItfv9FM7erUHqMKNkHHGban9iy6wzF"
        print(f"\nTesting access to folder: {folder_id}")
        result = syncer.test_folder_access(folder_id)
        print(f"Result: {result}")

    elif args.command == "test":
        if not syncer.authenticate():
            print("Please run 'setup' first to configure credentials")
            return

        if not syncer.setup_folder_structure():
            print("Failed to setup folder structure")
            return

        test_file = args.file
        if not os.path.exists(test_file):
            print(f"Error: Test file '{test_file}' does not exist")
            return

        print(f"Testing upload/update of: {test_file}")

        # Convert if it's a .py file
        if test_file.endswith(".py"):
            ipynb_file = syncer.convert_py_to_ipynb(test_file, force=True)
            if not ipynb_file:
                print("Failed to convert file")
                return
            upload_file = ipynb_file
        else:
            upload_file = test_file

        # Test upload
        folder_id = syncer.create_gdrive_folder_structure(
            test_file, syncer.gdrive_folder_id
        )
        success = syncer.upload_file(upload_file, folder_id)

        if success:
            print("✓ Test successful!")
        else:
            print("✗ Test failed!")

    elif args.command == "sync":
        paths = []

        if args.all:
            paths = ["notebooks"]
        elif args.folder:
            if not os.path.exists(args.folder):
                print(f"Error: Folder '{args.folder}' does not exist")
                return
            paths = [args.folder]
        elif args.file:
            if not os.path.exists(args.file):
                print(f"Error: File '{args.file}' does not exist")
                return
            paths = [args.file]

        syncer.sync_files(paths, args.dry_run, args.force)


if __name__ == "__main__":
    main()
