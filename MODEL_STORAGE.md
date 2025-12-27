# Model File Storage

The trained model file `pneumonia_detection_model.h5` (128MB) is **not included** in this repository due to GitHub's 100MB file size limit.

## Download the Model

You have two options to obtain the model:

### Option 1: Train Your Own Model
Follow the training instructions in the main medical project repository to generate `pneumonia_detection_model.h5`.

### Option 2: Use Git LFS (Recommended for Collaboration)

If you need to share the model via GitHub:

1. Install Git LFS:
   ```bash
   sudo apt-get install git-lfs  # Ubuntu/Debian
   brew install git-lfs          # macOS
   ```

2. Initialize Git LFS:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   ```

3. Add and commit the model:
   ```bash
   git add pneumonia_detection_model.h5
   git commit -m "Add model file with Git LFS"
   git push
   ```

### Option 3: External Storage

For deployment (Render, Heroku, etc.), you can:

1. **Upload to cloud storage** (Google Drive, Dropbox, AWS S3)
2. **Download during deployment** using a startup script
3. **Or manually upload** the model file to your hosting platform

## For Deployment on Render

When deploying to Render, upload the model file directly:

1. Deploy your code without the model file
2. Use Render's file upload feature or:
   - Add a build script to download from cloud storage
   - Or include the model via Git LFS

**Example build script** (if using cloud storage):
```bash
# In render.yaml, modify buildCommand:
buildCommand: |
  pip install -r requirements.txt &&
  wget https://your-storage-url/pneumonia_detection_model.h5 -O pneumonia_detection_model.h5
```

## Model Location

Place the downloaded `pneumonia_detection_model.h5` file in the **same directory** as `main.py` before running the API.

```
api/
├── main.py
├── pneumonia_detection_model.h5  ← Place model here
├── simple_viz.py
└── ...
```
