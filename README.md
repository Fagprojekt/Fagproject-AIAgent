
This is our repository for the "02122 Software Technology Project" course.
## Contents
- [Introduction](#Introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The project aims to make it easier to use the OceanWave framework by implementing an LLM-based agent that acts as a middleman between the user and OceanWave. The current capabilities of the agent include:

- Running a specific OceanWave simulation
- Visualizing the result of a simulation
- Listing the available files from certain folders
- Listing its own capabilities

## Installation

#### Windows

###### Prerequisites

- Ubuntu
- Docker (for running OceanWave example files)
- Python 3.11 or 3.12
- A fortran compiler
- Sparskit2
- LAPACK
- Harwell Subroutine Library (HSL)

###### Step-by-Step guide

- Clone the Fagproject-AIAgent repo and open it in Ubuntu
- Type in the following commands to prepare the python environment

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Add the .env file to the repo
- The rest of the commands are related to building OceanWave with docker and are covered in more detail at: [https://github.com/apengsigkarup/OceanWave3D-Fortran90/tree/botp/docker](https://github.com/apengsigkarup/OceanWave3D-Fortran90/tree/botp/docker)

Make sure the dependencies are in the project root directory.

Add a .env file with the following information (replaced by your own key):

OPENAI_API_KEY="Your-OpenAI-Key"

OPENAI_MODEL=gpt-4.1-nano


- Start the Docker engine
```
cd docker
chmod +x run_oceanwave3d.sh
cd ..
docker build -t docker_oceanwave3d .
```

#### Mac installation

Same as the Windows installation, but in a normal terminal rather than Ubuntu.

## Usage

To run the agent type:

`uvicorn src.ocean_agent.app:app --reload`

After when the web app is started the browser interface will be available locally at: http://127.0.0.1:8000/

The chat interface will be on the left and animated gifs will be available with the panel on the right.

The agent will then respond based on the prompt it receives. Certain guardrails are in place to keep the conversation on topic.

Examples of valid prompts are:

'Please list available input files'

'Run the NLStanding example'

'Visualize the results of the ObliqueRandomWave2D example'



