"""Tests for security audit parsing in sandbox runner."""

from sentinel.subagents.sandbox_runner import SandboxRunner


def test_parse_pip_audit_json():
    runner = SandboxRunner()
    sample_output = """{
        "dependencies": [
            {
                "name": "requests",
                "version": "2.28.0",
                "vulns": [
                    {
                        "id": "GHSA-xxxx-xxxx-xxxx",
                        "fix_versions": ["2.31.0"],
                        "aliases": ["CVE-2023-32681"],
                        "description": "Unintended leak of Proxy-Authorization header"
                    }
                ]
            }
        ]
    }"""
    result = runner._parse_pip_audit(sample_output)
    assert result.tool == "pip-audit"
    assert len(result.vulnerabilities) == 1
    assert result.vulnerabilities[0].package == "requests"
    assert result.vulnerabilities[0].severity == "high"


def test_parse_pip_audit_empty():
    runner = SandboxRunner()
    result = runner._parse_pip_audit('{"dependencies": []}')
    assert result.tool == "pip-audit"
    assert len(result.vulnerabilities) == 0
    assert "No vulnerabilities" in result.summary


def test_parse_npm_audit_json():
    runner = SandboxRunner()
    sample_output = """{
        "vulnerabilities": {
            "lodash": {
                "severity": "high",
                "via": [{"title": "Prototype Pollution", "url": "https://example.com"}],
                "fixAvailable": true
            }
        }
    }"""
    result = runner._parse_npm_audit(sample_output)
    assert result.tool == "npm-audit"
    assert len(result.vulnerabilities) == 1
    assert result.vulnerabilities[0].package == "lodash"
    assert result.vulnerabilities[0].severity == "high"


def test_parse_npm_audit_empty():
    runner = SandboxRunner()
    result = runner._parse_npm_audit('{"vulnerabilities": {}}')
    assert result.tool == "npm-audit"
    assert len(result.vulnerabilities) == 0
    assert "No vulnerabilities" in result.summary


def test_security_audit_result_to_dict():
    from sentinel.models.github import SecurityAuditResult, SecurityVulnerability

    result = SecurityAuditResult(
        tool="pip-audit",
        vulnerabilities=[
            SecurityVulnerability(
                package="flask",
                installed_version="2.3.0",
                fixed_version=None,
                severity="medium",
                advisory="CVE-2023-XXXX",
                url="https://example.com",
            )
        ],
        summary="Found 1 medium vulnerability",
        exit_code=1,
    )
    d = result.model_dump()
    assert d["tool"] == "pip-audit"
    assert len(d["vulnerabilities"]) == 1
