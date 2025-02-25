from ac.langs.decorator.reference import ReferenceDecorator


def Reference(link="Not descripted"):
    def wrap(func):
        return ReferenceDecorator(func, link)
    return wrap
