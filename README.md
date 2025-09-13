# SigmaRx7

SigmaRx7 is an open source synthetic healthcare ETL/ELT pipeline that reduces medication overlap, suggests lower‑cost generics and diversifies therapy based on the patient’s clinical history and payer rules. The system ingests HL7 v2 and FHIR data, normalizes prescriptions via RxNorm, enriches them with drug class and pricing data from open sources, detects therapeutic duplication, proposes generic substitutions, and applies a simple ML model to predict whether a recommendation would trigger a prior authorisation.

## Features

* **Ingestion & Normalisation** – Load synthetic HL7/FHIR patient data and standardise medications to RxNorm codes. A small DuckDB database stores patients, medications, conditions, allergies and coverage information for rapid query and analysis.
* **Drug Knowledge Base** – Join prescriptions to drug classes, Orange Book brand/generic links and public pricing/formulary data. These tables enable overlap detection and cost comparisons.
* **Recommendation Rules** – Built‑in rules identify duplicate therapies, suggest generic substitutes and align medications with the patient’s insurance formulary. A denial risk scoring function combines prior‑authorisation flags, tier placement and price to estimate payer acceptance.
* **FHIR Export** – Generate draft `MedicationRequest` resources for recommended generic substitutions, suitable for integration with downstream clinical systems.
* **Web UI** – A lightweight HTMX user interface allows you to select a patient and view recommendations in real‑time. A sample data set is provided via the bootstrap script.

## Project Structure

```
sigma-rx7/
  app/
    main.py              # FastAPI application and HTTP endpoints
    core/
      db.py             # Database connection helper
      models.py         # Pydantic models for API responses
    rules/
      overlap.py        # Therapeutic overlap detection
      generics.py       # Generic substitution suggestions
      payer.py          # Payer formulary lookups
    ml/
      denial.py         # Simple logistic scoring for denial risk
    fhir/
      export.py         # Draft MedicationRequest generator
    ui/
      templates/
        index.html      # HTMX front‑end
      static/
        styles.css      # Basic CSS styling
  scripts/
    bootstrap_duckdb.py # Initialise and seed the DuckDB database
  requirements.txt       # Python dependencies
  README.md              # Project overview and getting started
```

## Getting Started

1. **Install dependencies**

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Bootstrap the database**

Run the bootstrap script to initialise the DuckDB file (`sigma.duckdb`) and seed it with a small sample patient and medication set:

```bash
python scripts/bootstrap_duckdb.py
```

3. **Launch the web server**

Start the FastAPI application using Uvicorn. The `--reload` flag is helpful during development because it automatically reloads the server when you modify the code.

```bash
uvicorn app.main:app --reload
```

4. **View recommendations**

Open your browser and navigate to `http://127.0.0.1:8000`. Select the sample patient from the drop‑down and click **Run** to see overlap and generic substitution suggestions. The interface also lets you generate a draft FHIR `MedicationRequest` resource that shows how a generic switch could be represented in FHIR.

## Extending the Pipeline

* **Add more patients** – Update `scripts/bootstrap_duckdb.py` to insert additional rows into the `patients`, `conditions`, `meds` and other tables. See the script’s SQL statements for guidance.
* **Integrate real data** – Replace the synthetic seeding with your own HL7 v2 messages or FHIR bundles. Map HL7 segments or FHIR resources to the DuckDB schema, then run the recommendation logic. Use the `etl/loaders.py` module as a starting point.
* **Enhance ML scoring** – The `ml/denial.py` module currently contains a simple logistic‑style function based on price, tier and prior authorisation. You can replace this with a trained model using features derived from your data.

## License

This project is released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See `LICENSE` for details.
