from io import BytesIO
import csv
import re
import sys

from lxml import etree

from utils import (
    SentenceTokenizer,
    get_label,
    get_mdsd_domain_paths,
    get_mdsd_csv_file,
    CSV_FIELD_NAMES,
    FIELD_DOMAIN,
    FIELD_TEXT,
    FIELD_RATING,
    FIELD_LABEL,
    FIELD_PREPROCESSED,
)


def main():

    main_path = sys.argv[1]

    tokenizer = SentenceTokenizer()

    mdsd_paths = get_mdsd_domain_paths(main_path=main_path)
    mdsd_csv_path = get_mdsd_csv_file(main_path=main_path)

    rexp = re.compile('(<review>(.*?)</review>)', re.S)

    with open(mdsd_csv_path, mode='wt', encoding='utf-8', newline='') as wf:

        csvf = csv.DictWriter(wf, CSV_FIELD_NAMES, dialect='excel')
        csvf.writeheader()

        for mdsd_path in mdsd_paths:
            mdsd_path, mdsd_domain, mdsd_kind = mdsd_path
            print('Processing', mdsd_path, '... ', end='', flush=True)

            with open(mdsd_path, mode='rt', encoding='utf-8', newline='\n') as f:
                xml_docs = f.read()

                for m in rexp.finditer(xml_docs):
                    g = m.groups()[0]
                    ff = BytesIO(g.encode('utf-8'))
                    context = etree.iterparse(ff, recover=True)
                    mdsd_data = {}
                    for _, elem in context:
                        cnt = elem.text
                        if cnt is not None:
                            cnt = cnt.replace('\n', '')
                        else:
                            cnt = ''
                        mdsd_data[elem.tag] = cnt
                    try:
                        mdsd_review_text = mdsd_data['review_text']
                    except KeyError:
                        mdsd_review_text = ''
                    try:
                        rating = mdsd_data['rating']
                        rating = int(float(rating))
                    except KeyError:
                        rating = ''

                    row_content = {
                        FIELD_DOMAIN: mdsd_domain,
                        FIELD_TEXT: mdsd_review_text,
                        FIELD_RATING: rating,
                        FIELD_LABEL: get_label(mdsd_kind),
                    }

                    preprocessed = tokenizer.process(mdsd_review_text)
                    row_content[FIELD_PREPROCESSED] = (
                        ' '.join(list(preprocessed))
                    )

                    csvf.writerow(row_content)
            print('done')

    print('All done')


if __name__ == '__main__':
    main()
