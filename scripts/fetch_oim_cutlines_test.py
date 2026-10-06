from fetch_oim_cutlines import has_split_pages, page_key_for


def test_has_split_pages_needs_more_regions_than_documents() -> None:
    assert has_split_pages({"region_ct": "31", "document_ct": "19"})
    assert not has_split_pages({"region_ct": "19", "document_ct": "19"})
    assert not has_split_pages({"region_ct": "", "document_ct": "3"})


def test_page_key_for_falls_back_to_the_document_id() -> None:
    assert page_key_for({"id": 1, "title": "Champaign, Ill. | 1909 p12"}) == "p12"
    assert page_key_for({"id": 75469, "title": "Champaign, Ill. | 1909 index"}) == (
        "doc75469"
    )
