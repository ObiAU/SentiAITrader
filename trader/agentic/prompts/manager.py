from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateError, meta
from typing import List, Union, Literal

class PromptHandler:
    _env = None

    @classmethod
    def _get_env(cls, templates_dir="prompts/templates"):
        templates_dir = Path(__file__).parent.parent / templates_dir
        if cls._env is None:
            cls._env = Environment(
                loader=FileSystemLoader(templates_dir),
                undefined=StrictUndefined,
            )
        return cls._env


    @staticmethod
    def get_prompt(template, **kwargs):
        env = PromptHandler._get_env()
        template_path = f"{template}.j2"
        try:
            template_content = env.loader.get_source(env, template_path)[0]
            return env.from_string(template_content).render(**kwargs)
        except TemplateError as e:
            raise ValueError(f"Error rendering template: {str(e)}")


    @staticmethod
    def get_template_info(template):
        env = PromptHandler._get_env()
        template_path = f"{template}.j2"
        try:
            template_content = env.loader.get_source(env, template_path)[0]
            ast = env.parse(template_content)
            variables = meta.find_undeclared_variables(ast)
            return {
                "name": template,
                "variables": list(variables),
            }
        except Exception as e:
            raise ValueError(f"Error getting template info: {str(e)}")
