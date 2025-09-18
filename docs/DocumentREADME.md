## 📚 Sphinx Documentation

### How to Deploy to Read the Docs

1. **Create a Read the Docs account:**
   - Go to [readthedocs.org](https://readthedocs.org/) and sign up
   - Connect your GitHub/GitLab/Bitbucket account

2. **Import your project:**
   - Click "Import a Project" on your Read the Docs dashboard
   - Select your repository from the list
   - Configure the project name and settings

3. **Configure build settings:**
   - Ensure your `.readthedocs.yaml` file is in the repository root
   - Set the default branch (usually `main` or `master`)
   - Configure any environment variables if needed

4. **Build and deploy:**
   - Read the Docs will automatically build your documentation
   - Your docs will be available at `https://deepdatamininglearning.readthedocs.io/en/latest/`
   - Builds are triggered automatically on every commit to your default branch

5. **Custom domain (optional):**
   - Go to Admin → Domains in your Read the Docs project
   - Add your custom domain and configure DNS settings

### Alternative: GitHub Pages Hosting

Yes, Sphinx documentation can also be hosted on GitHub Pages as an alternative to Read the Docs:

#### Option 1: GitHub Pages

1. **Build documentation locally:**
   ```bash
   sphinx-build docs ./docs/build/html
   ```

2. **Create GitHub Pages workflow:**
   Create `.github/workflows/docs.yml`:
   ```yaml
   name: Build and Deploy Documentation
   on:
     push:
       branches: [ main ]
   jobs:
     build-and-deploy:
       runs-on: ubuntu-latest
       steps:
       - uses: actions/checkout@v3
       - name: Setup Python
         uses: actions/setup-python@v4
         with:
           python-version: '3.11'
       - name: Install dependencies
         run: |
           pip install sphinx sphinx-rtd-theme
       - name: Build documentation
         run: |
           sphinx-build docs ./docs/build/html
       - name: Deploy to GitHub Pages
         uses: peaceiris/actions-gh-pages@v3
         with:
           github_token: ${{ secrets.GITHUB_TOKEN }}
           publish_dir: ./docs/build/html
   ```

3. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Select "Deploy from a branch"
   - Choose "gh-pages" branch
   - Your docs will be available at `https://username.github.io/repository-name/`

**Alternative: Manual deployment without workflow**

If you prefer to build locally and push directly to GitHub Pages without using GitHub Actions:

1. **Build documentation locally:**
   ```bash
   sphinx-build docs ./docs/build/html
   ```

2. **Push built files to gh-pages branch:**
   ```bash
   # Navigate to the built documentation
   cd docs/build/html
   
   # Initialize git if not already done
   git init
   git add .
   git commit -m "Deploy documentation"
   
   # Push to gh-pages branch (replace with your repository URL)
   git remote add origin https://github.com/username/repository-name.git
   git push -f origin main:gh-pages
   ```

   Or use the `ghp-import` tool for easier deployment:
   ```bash
   # Install ghp-import
   pip install ghp-import
   
   # Build and deploy in one command
   sphinx-build docs ./docs/build/html
   ghp-import -n -p -f docs/build/html
   ```

#### Option 2: Host on Your Own Website

You can also build and deploy Sphinx documentation to your own web server or hosting service:

1. **Build documentation locally:**
   ```bash
   sphinx-build docs ./docs/build/html
   ```

2. **Deploy to your web server:**
   - **Via FTP/SFTP:** Upload the `./docs/build/html` folder contents to your web server
   - **Via SSH:** Use `rsync` or `scp` to transfer files:
     ```bash
     rsync -avz ./docs/build/html/ user@yourserver.com:/var/www/html/docs/
     ```
   - **Cloud hosting:** Upload to services like Netlify, Vercel, or AWS S3

3. **Automated deployment options:**
   - **CI/CD Pipeline:** Set up GitHub Actions, GitLab CI, or Jenkins to automatically build and deploy
   - **Webhook deployment:** Configure your server to pull and rebuild on git push
   - **Docker deployment:** Containerize your documentation with nginx

4. **Example nginx configuration:**
   ```nginx
   server {
       listen 80;
       server_name docs.yoursite.com;
       root /var/www/html/docs;
       index index.html;
       
       location / {
           try_files $uri $uri/ =404;
       }
   }
   ```

**Comparison:**
- **Read the Docs**: Better for open source projects, automatic builds, multiple format support
- **GitHub Pages**: Integrated with GitHub, custom workflows, free for public repositories
- **Own Website**: Full control, custom domain, can integrate with existing site, requires server management


### Quick Start - View Documentation

1. **Build the documentation:**
   ```bash
   ~/Documents/MyRepo/DeepDataMiningLearning/docs$ make html 
   (same to the above) sphinx-build -b html docs/source docs/build/html 
   
   /Documents/MyRepo/DeepDataMiningLearning/docs [0] $ python -m http.server 8000 -d build/html
   sphinx-build docs ./docs/build
   sphinx-autobuild docs/ docs/_build/html
   ```

2. **Serve locally:**
   ```bash
   cd docs/build
   python -m http.server 8000
   ```

3. **Open in browser:** http://localhost:8000/

### Documentation Setup & Dependencies

The documentation system has been upgraded with modern themes and full dependency resolution:

```bash
# Install required dependencies
pip install sphinx>=8.1.3 furo myst-parser nbsphinx nbsphinx-link sphinx-immaterial

# Build documentation
sphinx-build docs ./docs/build

# Check link integrity (optional)
sphinx-build docs -W -b linkcheck -d docs/build/doctrees docs/build/html
```

### Theme Configuration

The documentation uses the **Furo theme** - a modern, responsive theme with Material Design aesthetics:

```python
# In docs/conf.py
html_theme = 'furo'  # Modern, clean theme similar to Material Design
```

**Available themes:**
- `furo` - Current active theme (modern, Material Design-like)
- `sphinx_immaterial` - Direct adaptation of MkDocs-Material (installed, has config issues)
- `sphinx_rtd_theme` - Classic Read the Docs theme

### Supported File Formats

- **RestructuredText (.rst)** - Native Sphinx format
- **Markdown (.md)** - Via myst-parser extension
- **Jupyter Notebooks (.ipynb)** - Via nbsphinx extension

### Recent Fixes Applied

✅ **Resolved ImportError with Python 3.11+**: Upgraded Sphinx from 3.5.3 to 8.1.3  
✅ **Added missing extensions**: myst-parser, nbsphinx-link  
✅ **Modern theme integration**: Furo theme with Material Design aesthetics  
✅ **Dependency compatibility**: All extensions now work together seamlessly  

The generated HTML files are in `docs/build/`. You can also view the documents at: [readthedocs](https://deepdatamininglearning.readthedocs.io/en/latest/)
