from icrawler.builtin import GoogleImageCrawler

NUM_PER_QUERY = 100


def downloadQuery(query):
    google_crawler = GoogleImageCrawler(storage={'root_dir': f'anthurium_images_og/{query}'})
    google_crawler.crawl(keyword=query+" -x", max_num=NUM_PER_QUERY)


def downloadQueries(queries):
    for query in queries:
        downloadQuery(query)

ANTHURIUM_VARIETIES_ALL = ["ANTHURIUM VEITCHII (KING)","ANTHURIUM WAROQUEANUM (QUEEN)","ANTHURIUM LUXURIANS","ANTHURIUM SPLENDIDUM","ANTHURIUM CORRUGATUM","ANTHURIUM RUGULOSUM","ANTHURIUM REGALE","ANTHURIUM CLARINERVIUM ","ANTHURIUM CRYSTALLINUM JUVENILE ","ANTHURIUM CRYSTALLINUM CRYSTAL HOPE ","ANTHURIUM CRYSTALLINUM SILVER BUSH","ANTHURIUM FORGETII","ANTHURIUM FORGETII WHITE STRIPES","ANTHURIUM BESSEAE ","ANTHURIUM ‘ACE OF SPADES’","ANTHURIUM VILLENAORUM ","ANTHURIUM ARGYROSTACHYUM VILLOSUM","ANTHURIUM BALAOANUM","ANTHURIUM POLYNEURON","ANTHURIUM LINEOLATUM","ANTHURIUM REFLEXINERVIUM","ANTHURIUM PELTIGERUM","ANTHURIUM NERVATUM","ANTHURIUM WATERBURNAYUM","ANTHURIUM WAROCQUEANUM X WATERBURNAYUM","ANTHURIUM BULLATUS","ANTHURIUM DEBILIS","ANTHURIUM VITTARIFOLIUM","ANTHURIUM PALLIDIFLORUM","ANTHURIUM WENDLINGERI","ANTHURIUM BAKERI","ANTHURIUM OVATIFOLIUM","ANTHURIUM OVATIFOLIUM LONG","ANTHURIUM ECUADORENSIS","ANTHURIUM ANGAMARCANUM","ANTHURIUM MAGNIFICUM X CRYSTALLINUM","ANTHURIUM RADICANS","ANTHURIUM RADICANS X DRESSLERI","ANTHURIUM MOODEANUM","ANTHURIUM LENTII","ANTHURIUM OXACA","ANTHURIUM ARROW","ANTHURIUM MAGNIFICUM","ANTHURIUM WATERMALIENSE","ANTHURIUM SUPERBUM","ANTHURIUM BLACK DRAGON","ANTHURIUM PLOWMANII","ANTHURIUM RENAISSANCE","ANTHURIUM OXYCARPUM","ANTHURIUM JENMANII","ANTHURIUM GOLDEN JENMANII","ANTHURIUM JENMANII VARIEGATED","ANTHURIUM JUNGLE KING","ANTHURIUM GRACILE","ANTHURIUM ELLIPTICUM ‘JUNGLE BUSH’","ANTHURIUM FAUSTINO’S GIANT","ANTHURIUM MACROLOBIUM","ANTHURIUM MACROLOBIUM X PEDATORADIATUM","ANTHURIUM PEDATORADIATUM","ANTHURIUM PSEUDOCLAVIGERUM","ANTHURIUM ZIPPELIANUM","ANTHURIUM CLAVIGERUM","ANTHURIUM POLYSCHISTUM","ANTHURIUM TRIPHYLLUM","ANTHURIUM PODOPHYLLUM","ANTHURIUM PEDATUM","ANTHURIUM PEDATORADIATUM","ANTHURIUM ANDREANUM SIERRA RED ","ANTHURIUM ANDREANUM SIERRA WHITE ","ANTHURIUM ANDREANUM PRINCESS AMALIA","ANTHURIUM WHITE HEART","ANTHURIUM WHITE WINNER","ANTHURIUM PINK EXPLOSION","ANTHURIUM BLACK LOVE","ANTHURIUM ERNESTII","ANTHURIUM SAFARI","ANTHURIUM SIMBA","ANTHURIUM LUMINA","ANTHURIUM GIGANTEUM","ANTHURIUM SALGARENSE","ANTHURIUM FAUSTOMIRANDAE"]

ANTHURIUM_VARIETIES_SAMPLE = ["ANTHURIUM CRYSTALLINUM", "ANTHURIUM WAROQUEANUM (QUEEN)", "ANTHURIUM LUXURIANS", "ANTHURIUM REGALE", "ANTHURIUM WENDLINGERI"]

downloadQueries(ANTHURIUM_VARIETIES_SAMPLE)