import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let extension = {
    name: "FairLab",
    nodeCreated(node, app) {
        if (node?.comfyClass === "DownloadImageClass") {
            node.onExecuted = function (output) {
                let imgURLs = [];
                if (output?.images) {
                    imgURLs = imgURLs.concat(
                        output.images.map((params) => {
                            return api.apiURL("/view?" + new URLSearchParams(params).toString() + (this.animatedImages ? "" : app.getPreviewFormatParam()) + app.getRandParam());
                        })
                    );
                }

                if (imgURLs.length > 0) {
                    Promise.all(
                        imgURLs.map((src) => {
                            return new Promise((r) => {
                                const img = new Image();
                                img.onload = () => r(img);
                                img.onerror = () => r(null);
                                img.src = src;
                            });
                        })
                    ).then((imgs) => {
                        function download_images(imgs) {
                            for (let i = 0; i < imgs.length; i++) {
                                let img = imgs[i];
                                let a = document.createElement("a");
                                let url = new URL(img.src);
                                url.searchParams.delete("preview");
                                a.href = url;
                                a.setAttribute("download", new URLSearchParams(url.search).get("filename"));
                                document.body.append(a);
                                a.click();
                                requestAnimationFrame(() => a.remove());
                            }
                        }

                        if (imgs) download_images(imgs);
                    });
                }
            };
        }
    },

    async setup() {
        //old ui
        const restartButton = document.createElement("button");
        restartButton.textContent = "Restart";
        restartButton.title = "Restart the server";
        restartButton.onclick = () => {
            api.fetchApi("/manager/reboot");
        };

        const menu = document.querySelector(".comfy-menu");
        if (menu) menu.appendChild(restartButton);

        //new ui
        const comfyui_button_restart = document.createElement("button");
        comfyui_button_restart.textContent = "Restart";
        comfyui_button_restart.className = "comfyui-button";
        comfyui_button_restart.title = "Restart the server";
        comfyui_button_restart.onclick = () => {
            api.fetchApi("/manager/reboot");
        };
        var comfyui_menu_group = document.querySelector(".comfyui-button-group");
        if (comfyui_menu_group) comfyui_menu_group.appendChild(comfyui_button_restart);
    },
};

app.registerExtension(extension);
