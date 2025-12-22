# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "bsp_tool",
#     "dearpygui",
#     "numpy",
# ]
# ///

# https://dearpygui.readthedocs.io/en/latest/index.html


from typing import Any, Dict, List

import dearpygui.dearpygui as imgui
import numpy as np

import bsp_tool


# TODO: free texture when deleted
class Viewer:
    bsp: bsp_tool.base.Bsp
    tags: Dict[str, Any]

    def __init__(self):
        self.bsp = None
        # TODO: HDR equation controls (sliders & data)
        self.data = {
            "face_index": 0,
            "ldr": False,
            "hdr": False}
        self.tags = {
            "image": None,
            "slider.face": None,
            "texture": None}

    def face_texels(self) -> np.array:
        face = self.bsp.FACES[self.data["face_index"]]
        if face.light_offset == -1 or face.styles == -1:
            return None  # face is not lightmapped
        width, height = map(int, [s + 1 for s in face.lightmap.size])
        start, length = face.light_offset, width * height * 4
        # TODO: UI controls to select LDR/HDR + A/B/C/D + style index
        # -- offsets = [start + (length * i) for i in range(4)]  # ABCD
        # -- if out.data["hdr"]: lump = self.bsp.LIGHTING_HDR
        # -- if out.data["ldr"]: lump = self.bsp.LIGHTING
        texels = bytes(self.bsp.LIGHTING[start:start + length])  # LDR.A
        out = (np.frombuffer(texels, dtype=np.uint8) / 255).astype(np.float32)
        return out.reshape((width, height, 4))

    def pixels(self) -> np.array:
        width, height = 128, 128
        texture_floats = (np.frombuffer(
            b"\x00\x00\x00\x00" * width * height,
            dtype=np.uint8) / 255).astype(np.float32)
        # layer lightmap texels over base texture
        texture_floats = texture_floats.reshape(width, height, 4)
        sub_image = self.face_texels()
        if sub_image is not None:
            # TODO: HDR exponent scaling
            # -- RGB = RGB * (A * 255)
            # -- remap_to_ldr(RGB, ...)
            # -- A = 1.0
            sub_width, sub_height = sub_image.shape[:2]
            assert sub_width <= 128
            assert sub_height <= 128
            texture_floats[:sub_width, :sub_height] = sub_image
        else:
            # TODO: update a UI indicator for no lightmap available
            ...
        return texture_floats.flatten()

    def update(self):
        """update texture to reflect current index"""
        imgui.set_value(self.tags["texture"], self.pixels())

    # callbacks
    def face_callback(self):
        self.data["face_index"] = imgui.get_value(self.tags["slider.face"])
        self.update()

    @classmethod
    def from_dialog(cls, sender: str, app_data: Dict[str, Any], parent: str):
        name, path = list(app_data["selections"].items())[0]
        return cls.from_path(path, parent)

    @classmethod
    def from_path(cls, path: str, parent: str):
        out = cls()
        out.bsp = bsp_tool.load_bsp(path)
        # assert out.bsp.branch == bsp_tool.branches.nexon.cso2_2018
        # state checks
        has_ldr = bool(out.bsp.headers["LIGHTING"].length > 0)
        has_hdr = bool(out.bsp.headers["LIGHTING_HDR"].length > 0)
        out.data.update({"ldr": has_ldr, "hdr": has_hdr})
        # register texture
        # NOTE: 128x128 is a hardcoded max, lightmap shouldn't exceed it
        with imgui.texture_registry(show=False):
            width, height = 128, 128
            texture_floats = (np.frombuffer(
                b"\xFF\x00\xFF\xFF" * width * height,
                dtype=np.uint8) / 255).astype(np.float32)
            out.tags["texture"] = imgui.add_raw_texture(
                width=width, height=height,
                default_value=texture_floats,
                format=imgui.mvFormat_Float_rgba)
        # UI layout
        with imgui.child_window(parent=parent):
            with imgui.group(horizontal=True):
                with imgui.group(width=192):
                    out.tags["slider.face"] = imgui.add_slider_int(
                        label="Face",
                        min_value=0,
                        max_value=len(out.bsp.FACES) - 1,
                        callback=out.face_callback)
                    # TODO: HDR equation controls
                with imgui.group():
                    width, height = [128 * 4, 128 * 4]
                    out.tags["image"] = imgui.add_image(
                        out.tags["texture"],
                        width=width, height=height)
                    # TODO: zoom
                    # -- on scroll
                    # -- slider
        # load face lightmap texture
        out.update()
        return out


def main(*args: List[str]):
    """create an imgui loader with a file browser"""
    imgui.create_context()

    # file browser config & callback
    with imgui.file_dialog(
            directory_selector=False,
            show=False,
            callback=lambda s, a: Viewer.from_dialog(s, a, "main"),
            tag="file_browser",
            width=768, height=320):
        # NOTE: case-sensitive, idk why
        imgui.add_file_extension("CS:O2 Map (*.bsp){.bsp}")

    # NOTE: forcing 4k min size to ensure we fill the viewport
    with imgui.window(tag="main", min_size=(4096, 4096)):
        with imgui.menu_bar():
            imgui.add_menu_item(
                label="Open",
                callback=lambda: imgui.show_item("file_browser"))

    imgui.create_viewport(title="CS:O2 Lightmap Viewer", width=512, height=512)
    imgui.setup_dearpygui()
    imgui.show_viewport()
    imgui.set_primary_window("main", True)

    # open a file from args
    if len(args) == 1:
        Viewer.from_path(args[0], "main")

    imgui.start_dearpygui()

    imgui.destroy_context()


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
