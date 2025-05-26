# Google Drive Sync Setup Guide

This guide will help you set up the Google Drive sync functionality to upload your notebooks automatically to Google Drive for use with Google Colab.

## Quick Start

1. **Run the setup command:**
   ```bash
   ./sync_to_gdrive.sh setup
   ```

2. **Check configuration status:**
   ```bash
   ./sync_to_gdrive.sh status
   ```

3. **Sync all notebooks:**
   ```bash
   ./sync_to_gdrive.sh sync --all
   ```

## Detailed Setup Process

### Step 1: Enable Google Drive API

The setup script will guide you through this process, but here are the detailed steps:

1. Go to [Google Cloud Console](https://console.developers.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click on it and press "Enable"

### Step 2: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in the required information (app name, user support email, etc.)
   - Add your email to test users
4. For application type, choose "Desktop application"
5. Give it a name (e.g., "Geomatica Notebooks Sync")
6. Click "Create"
7. Download the JSON file
8. Save it as `credentials.json` in the project root directory

### Step 3: Run Setup

```bash
./sync_to_gdrive.sh setup
```

This will:
- Check if `credentials.json` exists
- Open a browser window for authentication
- Save your authentication token for future use
- Configure your target Google Drive folder

**Folder Configuration Options:**
1. **Use existing folder**: Provide a folder ID or Google Drive URL
2. **Create new folder**: Specify a name (default: "Geomatica-Aplicada-Notebooks")

### Step 4: Start Syncing

```bash
# Sync all notebooks
./sync_to_gdrive.sh sync --all

# Or test first with dry-run
./sync_to_gdrive.sh sync --all --dry-run
```

## Usage Examples

### Sync Commands

```bash
# Setup (only needed once)
./sync_to_gdrive.sh setup

# Force refresh credentials (if having issues)
./sync_to_gdrive.sh setup --force-refresh

# Check current configuration
./sync_to_gdrive.sh status

# Reconfigure folder (if needed)
./sync_to_gdrive.sh config

# Reset stored credentials
./sync_to_gdrive.sh reset

# Sync all notebooks
./sync_to_gdrive.sh sync --all

# Sync specific folder
./sync_to_gdrive.sh sync --folder notebooks/01_vector

# Sync specific file
./sync_to_gdrive.sh sync --file notebooks/01_vector/01_introduccion_datos_vectoriales.py

# Preview what would be synced (dry run)
./sync_to_gdrive.sh sync --all --dry-run

# Force reconversion of all files
./sync_to_gdrive.sh sync --all --force
```

### Python Commands (Alternative)

You can also use the Python script directly:

```bash
# Setup
python sync_to_gdrive.py setup

# Setup with force refresh
python sync_to_gdrive.py setup --force-refresh

# Check status
python sync_to_gdrive.py status

# Configure folder
python sync_to_gdrive.py config

# Reset credentials
python sync_to_gdrive.py reset

# Sync all
python sync_to_gdrive.py sync --all

# Dry run
python sync_to_gdrive.py sync --all --dry-run
```

## What Happens During Sync

1. **File Discovery**: The script finds all `.py` files in the specified path(s)
2. **Conversion**: Converts `.py` files (Jupytext format) to `.ipynb` using `jupytext`
3. **Folder Creation**: Creates folder structure in Google Drive matching your local structure
4. **Upload**: Uploads the `.ipynb` files to Google Drive
5. **Update Check**: Only uploads files that have been modified (efficient)

## Using Existing Google Drive Folders

### Getting a Folder ID

If you want to use an existing folder in Google Drive:

1. Open Google Drive in your browser
2. Navigate to the folder you want to use
3. Look at the URL in the address bar:
   ```
   https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74mMjoeqXVE
   ```
4. The folder ID is the long string after `folders/`: `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74mMjoeqXVE`

### Configuration Options

During setup, you can choose:
- **Option 1**: Use existing folder (provide folder ID or URL)
- **Option 2**: Create a test folder to verify access
- **Option 3**: Create new folder (specify name)

The script will remember your choice and use the same folder for future syncs.

### Test Folder Feature

If you're having trouble with folder access, you can create a test folder:
- The script creates a folder named "Test-Folder-Geomatica-Aplicada-Notebooks"
- You get the folder ID immediately
- You can verify access works correctly
- You can rename or delete the folder later if needed

## Credential Management

### Smart Credential Handling

The script intelligently manages your Google Drive credentials:

1. **First Setup**: Downloads and stores authentication token
2. **Subsequent Runs**: Reuses existing valid token
3. **Automatic Refresh**: Refreshes expired tokens automatically
4. **Force Refresh**: Option to force new authentication

### Credential Commands

```bash
# Check if credentials are working
./sync_to_gdrive.sh status

# Force refresh credentials (useful if having auth issues)
./sync_to_gdrive.sh setup --force-refresh

# Reset all stored credentials
./sync_to_gdrive.sh reset
```

### Configuration Files

The script creates these files (automatically added to .gitignore):
- `credentials.json` - OAuth2 credentials from Google Cloud Console
- `token.pickle` - Stored authentication token
- `gdrive_config.json` - Folder configuration

**Example gdrive_config.json:**
```json
{
  "folder_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74mMjoeqXVE",
  "folder_name": "Geomatica-Aplicada-Notebooks"
}
```

## Folder Structure in Google Drive

The script creates/uses a folder in your Google Drive with the same structure as your local repository:

```
Geomatica-Aplicada-Notebooks/
├── 00_intro.ipynb
├── 01_vector/
│   └── 01_introduccion_datos_vectoriales.ipynb
└── 02_raster/
    ├── 01_datos_raster.ipynb
    ├── 02_intro_teledeteccion.ipynb
    └── 03_acceso_imagenes.ipynb
```

## Using with Google Colab

Once synced, you can access your notebooks in Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" > "Open notebook"
3. Select "Google Drive" tab
4. Navigate to `Geomatica-Aplicada-Notebooks` folder
5. Open any notebook

## Troubleshooting

### Error: credentials.json not found

Make sure you've downloaded the OAuth 2.0 credentials file and saved it as `credentials.json` in the project root.

### Error: The file credentials.json does not exist

Double-check the filename and location. It should be exactly `credentials.json` in the same directory as the sync scripts.

### Folder Access Issues

If you get "Folder not found" or access errors:

**Common causes:**
- Folder ID is incorrect or incomplete
- Folder was deleted or moved
- You don't have permission to access the folder
- Folder is in a different Google account
- **Folder is in a shared drive (Team Drive)** with restricted access

**Solutions:**
1. **Verify the folder ID**: Make sure you copied the complete folder ID from the URL
2. **Check permissions**: Ensure the folder is shared with your Google account
3. **Shared drive access**: If the folder is in a shared drive, ensure you have proper access
4. **Try test folder**: Use option 2 during setup to create a test folder
5. **Use correct account**: Make sure you're authenticated with the right Google account

### Shared Drive (Team Drive) Support

The script now supports folders in shared drives. If your folder is in a shared drive:
- Make sure you have access to the shared drive
- Verify you can see the folder in Google Drive web interface
- The folder ID works the same way as personal drive folders
- **Important**: If you previously set up credentials, you may need to reset them to get the new permissions

**If you're having shared drive issues:**
```bash
# Reset credentials to get new permissions
./sync_to_gdrive.sh reset

# Setup again with broader permissions
./sync_to_gdrive.sh setup
```

**Getting the correct folder ID:**
```
URL: https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74mMjoeqXVE
ID:  1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74mMjoeqXVE
```

### Authentication errors

If you get authentication errors:
1. Delete `token.pickle` file
2. Run the setup command again: `./sync_to_gdrive.sh setup`

### Permission errors

Make sure you have the correct permissions:
- The OAuth app has access to Google Drive
- Your Google account has permission to access the files

### Rate limiting

If you encounter rate limiting errors:
- Wait a few minutes and try again
- Sync smaller batches of files instead of all at once

## Security Notes

- `credentials.json` and `token.pickle` are automatically added to `.gitignore`
- Never commit these files to version control
- The app requests these Google Drive permissions:
  - `drive.file`: Access to files created by the app
  - `drive`: Broader access needed for shared drives and existing folders
- You can revoke access anytime from your Google Account settings

## Dependencies

The script automatically installs these Python packages:
- `jupytext>=1.14.0` - For converting .py to .ipynb
- `google-api-python-client>=2.0.0` - Google Drive API client
- `google-auth-httplib2>=0.1.0` - HTTP transport for Google Auth
- `google-auth-oauthlib>=0.5.0` - OAuth2 flow for Google APIs

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Try running with `--dry-run` first to see what would happen
3. Ensure all dependencies are installed
4. Check that your Google Cloud project has the Drive API enabled 