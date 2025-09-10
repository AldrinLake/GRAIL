def get_model(model_name, args):
    name = model_name.lower()
    if name == "grail":
        from models.GRAIL.GRAIL import GRAIL
        return GRAIL(args)
    else:
        assert 0
