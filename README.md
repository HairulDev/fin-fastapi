# Project Documentation

This repository contains documentation and references for both the **backend** (.NET Core 8) and **backend** (Rust Axum for upload/download file) 
and **backend** (Python FastAPI) and **frontend** (Vue.js or React.js or Angular.js)

---

## Demo

Watch the demo on YouTube: [https://youtu.be/F95vnIzb4mU?feature=shared]

## Related Projects

- **.NET Core 8 (Backend)**: [fin-netcore](https://github.com/HairulDev/fin-netcore)
- **Vue.js (Frontend)**: [fin-vuejs](https://github.com/HairulDev/fin-vuejs)
- **Rust (Axum) Version**: [fin-rustaxum](https://github.com/HairulDev/fin-rustaxum)
- **React (TypeScript) Version**: [fin-reactts](https://github.com/HairulDev/fin-reactts)
- **Angular (TypeScript) Version**: [fin-angular](https://github.com/HairulDev/fin-angular)
- **Python FastAPI Version**: [fin-fastapi](https://github.com/HairulDev/fin-fastapi)

---

## Prerequisites
- Python 3.10 or newer
- [pip](https://pip.pypa.io/) (Python package manager)
- [virtualenv](https://virtualenv.pypa.io/)

## Installation
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a `.env` file in the project root directory and add the necessary environment variables:

```env
API_FMP=https://financialmodelingprep.com
FMP_API_KEY=your_api_key (go to https://financialmodelingprep.com)
```

## Running the Project

```bash
# Active virtual environment
python3.12 -m venv env
source env/bin/activate
```


```bash
# Training
train/train_lstm.py
```

```bash
# Start the development server.
uvicorn main:app --host 127.0.0.1 --port 9000 --reload
```

