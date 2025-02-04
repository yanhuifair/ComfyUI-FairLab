import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { $el } from "../../../scripts/ui.js";

let extension = {
    name: "FairLab",
    nodeCreated(node, app) {
        if (node?.comfyClass === "DownloadImageNode") {
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
        let restartButton;

        //old ui
        const menu = document.querySelector(".comfy-menu");
        if (menu) {
            restartButton = document.createElement("button");
            restartButton.textContent = "Restart";
            restartButton.tooltip = "Restart the server";
            restartButton.onclick = () => {
                api.fetchApi("/manager/reboot");
            };
            menu.appendChild(restartButton);
        }

        //new ui
        if (!app.menu?.element.style.display && app.menu?.settingsGroup) {
            restartButton = new (await import("../../../scripts/ui/components/button.js")).ComfyButton({
                icon: "restart",
                action: () => {
                    api.fetchApi("/manager/reboot");
                },
                tooltip: "Restart the server",
                content: "Restart",
            });
            restartButton.enabled = true;
            restartButton.element.style.display = "";
            app.menu.settingsGroup.append(restartButton);
        }
    },
};

app.registerExtension(extension);
