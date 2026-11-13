from dta_project.audit_overlap import overlap_report


def test_cross_dataset_exposure_is_reported(smoke_root):
    report = overlap_report(
        smoke_root / "pcba",
        smoke_root / "pcba/splits/random_seed42.json",
        smoke_root / "dta",
        smoke_root / "dta/splits/cold_drug_seed42.json",
    )
    assert report["pcba_training_molecules"] == 16
    assert "drugs_in_pcba_train" in report["downstream"]["test"]
