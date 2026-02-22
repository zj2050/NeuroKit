import csv
import operator
import os

from docutils import nodes
from docutils.parsers.rst import Directive


MIN_FIELD_LENGTH = 2

abrv_to_sensor = {
    "ecg": "Electrocardiography",
    "eda": "Electrodermal Activity",
    "rsp": "Respiration",
    "ppg": "Photoplethysmography",
    "eeg": "Electroencephalography",
    "emg": "Electromyography",
    "eog": "Electrooculography",
    "hrv": "Heart Rate Variability",
}


class CSVDocDirective(Directive):
    has_content = True

    def run(self):
        env = self.state.document.settings.env

        # We use a dictionary keyed by docname to make parallel merging easy
        if not hasattr(env, "nk_codebook_storage"):
            env.nk_codebook_storage = {}

        docname = env.docname
        if docname not in env.nk_codebook_storage:
            env.nk_codebook_storage[docname] = []

        doc_source_name = env.temp_data.get("object", "unknown")
        maybe_sensor = doc_source_name.split("_")
        doc_sensor = abrv_to_sensor.get(maybe_sensor[0], "N/A") if maybe_sensor else "N/A"

        bullet_list = nodes.bullet_list()

        for line in self.content:
            fields = [f.strip() for f in line.split("|")]
            if len(fields) < MIN_FIELD_LENGTH:
                continue

            clean_fields = [" ".join(f.split()) for f in fields]

            # Store data in the environment dictionary for this specific file
            env.nk_codebook_storage[docname].append([clean_fields[0], clean_fields[1], doc_sensor, f"{doc_source_name}.py"])

            # Prepare the UI nodes for the HTML page
            paragraph = nodes.paragraph()
            paragraph += nodes.literal("", "", nodes.Text(clean_fields[0]))
            paragraph += nodes.Text(": ")
            paragraph += nodes.Text(clean_fields[1])

            list_item = nodes.list_item()
            list_item += paragraph
            bullet_list += list_item

        return [bullet_list]


def on_env_purge_doc(app, env, docname):
    """Prevents duplicate entries when a file is edited and re-read."""
    if hasattr(env, "nk_codebook_storage"):
        env.nk_codebook_storage.pop(docname, None)


def on_env_merge_info(app, env, docnames, other):
    """Merges data from different CPU cores back into the main process."""
    if not hasattr(env, "nk_codebook_storage"):
        env.nk_codebook_storage = {}
    if hasattr(other, "nk_codebook_storage"):
        env.nk_codebook_storage.update(other.nk_codebook_storage)


def write_codebook_to_csv(app, exception):
    """Final step: Flatten, Sort, and Write to CSV."""
    if exception or not hasattr(app.env, "nk_codebook_storage"):
        return

    # Flatten the dictionary of lists into one list
    all_data = []
    for doc_data in app.env.nk_codebook_storage.values():
        all_data.extend(doc_data)

    if not all_data:
        return

    # Sort Codebook by sensor
    all_data.sort(key=operator.itemgetter(2, 0))

    csv_file_path = os.path.join(app.outdir, "_static", "neurokit_codebook.csv")
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # 4. Write to file
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Field Name", "Field Description", "Field Category", "Source File Name"])
        writer.writerows(all_data)

    print(f"\nCodebook: {len(all_data)} items saved to {csv_file_path}")


def setup(app):
    app.add_directive("codebookadd", CSVDocDirective)

    app.connect("env-purge-doc", on_env_purge_doc)
    app.connect("env-merge-info", on_env_merge_info)
    app.connect("build-finished", write_codebook_to_csv)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
