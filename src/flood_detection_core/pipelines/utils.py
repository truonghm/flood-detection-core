import datetime


def get_pretrain_run_name() -> str:
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pretrain_{current_date}"
    return run_name


def get_site_specific_run_name(site_name: str) -> str:
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"site_specific_{site_name}_{current_date}"
    return run_name
