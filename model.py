import timm


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model