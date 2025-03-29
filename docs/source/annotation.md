
# Label Studio
Label Studio is an open-source data labeling tool. It's designed to help you label various types of data (images, text, audio, etc.) for machine learning model training.

https://labelstud.io/guide/install

Poetry helps you declare, manage, and install the libraries your Python project depends on. It ensures you have the correct versions of those libraries. It replaces older methods that could be cumbersome, like managing requirements.txt files. Poetry also assists in packaging your Python project for distribution. This means it helps you create packages that others can easily install and use.

Uses pyproject.toml for project configuration. The poetry install command is crucial for setting up your project's environment.
* Generates a poetry.lock file to ensure consistent installations across different environments. 
* The poetry.lock file plays a crucial role here. It stores the exact versions of all installed packages, ensuring consistent installations across different environments.


A fundamental pyproject.toml file will typically include these sections:
* [tool.poetry]: This section contains Poetry-specific settings, such as your project's name, version, and dependencies.
* [build-system]: This section defines the build system requirements.
* [project]: This section contains core metadata about your project.

`https://github.com/HumanSignal/label-studio/blob/develop/label_studio/__init__.py`: This file signifies that the label_studio directory is a Python package. 
* In Python, the `__init__.py` file serves two primary purposes: 1) Package Marker: Its presence in a directory tells Python that the directory should be treated as a Python package; 2) It can be used to execute initialization code when the package is imported. It can also be used to control what is imported when someone uses `from xx import *`.
* `importlib.metadata` is a module in the Python standard library (since Python 3.8) that provides tools for accessing metadata about installed Python packages. It allows you to retrieve information like the package's version, dependencies, entry points


```bash
git clone https://github.com/HumanSignal/label-studio.git

# install dependencies
cd label-studio
pip install poetry
poetry install

# run db migrations
# poetry run tells Poetry to execute a command within the project's virtual environment.
poetry run python label_studio/manage.py migrate

# collect static files
poetry run python label_studio/manage.py collectstatic

# start the server in development mode at http://localhost:8080
poetry run python label_studio/manage.py runserver
#make it accessible externally, you need to bind it to 0.0.0.0, which means it will listen on all available network interfaces.
poetry run python label_studio/manage.py runserver 0.0.0.0:8080
#create account lkk688@gmail.com Liu123456
#go to example folder:
docker-compose up
#Open your docker-compose.yml file
#Add the environment section to the service definition for your ML backend. Include the LABEL_STUDIO_API_KEY variable with your API key.
#change LABEL_STUDIO_HOST address
docker-compose down
docker-compose up --build
```


```bash
npx create-react-app my-labeling-app
cd my-labeling-app
npm install react react-dom
git clone https://github.com/HumanSignal/label-studio.git
cp -r label-studio/web/libs/editor src/label-studio-editor
cd src/label-studio-editor
npm install
cd../../
# create src/App.js
npm start
#When you're ready to deploy your application, you'll need to create a production build using npm run build.
```

Import and use the editor component:
```bash
// src/App.js
import React from 'react';
import LabelStudio from './label-studio-editor';

function App() {
  return (
    <div>
      <LabelStudio />
    </div>
  );
}

export default App;
```

