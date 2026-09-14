import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SignUp from '../../pages/SignUp';
import { api } from '../../services/api';

vi.mock('../../services/api');

describe('SignUp Page', () => {
  const onLogin = vi.fn();
  const onSwitchToLogin = vi.fn();

  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders sign-up form correctly', () => {
    render(<SignUp onLogin={onLogin} onSwitchToLogin={onSwitchToLogin} />);
    expect(screen.getByLabelText('Full Name')).toBeInTheDocument();
    expect(screen.getByLabelText('Email Address')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
    expect(screen.getByLabelText('Confirm Password')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Account' })).toBeInTheDocument();
  });

  it('updates input fields on user input', async () => {
    render(<SignUp onLogin={onLogin} onSwitchToLogin={onSwitchToLogin} />);
    await userEvent.type(screen.getByLabelText('Full Name'), 'Test User');
    await userEvent.type(screen.getByLabelText('Email Address'), 'test@example.com');
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.type(screen.getByLabelText('Confirm Password'), 'password123');
    expect(screen.getByLabelText('Full Name')).toHaveValue('Test User');
    expect(screen.getByLabelText('Email Address')).toHaveValue('test@example.com');
    expect(screen.getByLabelText('Password')).toHaveValue('password123');
    expect(screen.getByLabelText('Confirm Password')).toHaveValue('password123');
  });

  it('shows error on password mismatch', async () => {
    const { container } = render(<SignUp onLogin={onLogin} onSwitchToLogin={onSwitchToLogin} />);
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.type(screen.getByLabelText('Confirm Password'), 'password456');
    const form = container.querySelector('form');
    fireEvent.submit(form!);
    await waitFor(() => {
      expect(screen.getByText('Passwords do not match')).toBeInTheDocument();
    });
    expect(api.register).not.toHaveBeenCalled();
  });

  it('calls onLogin on successful registration', async () => {
    (api.register as vi.Mock).mockResolvedValueOnce({ access_token: 'test-token' });
    const { container } = render(<SignUp onLogin={onLogin} onSwitchToLogin={onSwitchToLogin} />);
    await userEvent.type(screen.getByLabelText('Full Name'), 'Test User');
    await userEvent.type(screen.getByLabelText('Email Address'), 'test@example.com');
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.type(screen.getByLabelText('Confirm Password'), 'password123');
    const form = container.querySelector('form');
    fireEvent.submit(form!);
    // Wait for the async registration and login call
    await waitFor(() => {
      expect(api.register).toHaveBeenCalledWith('test@example.com', 'Test User', 'password123');
      expect(onLogin).toHaveBeenCalled();
    });
  });

  it('shows error on failed registration', async () => {
    (api.register as vi.Mock).mockRejectedValueOnce(new Error('Email already exists'));
    const { container } = render(<SignUp onLogin={onLogin} onSwitchToLogin={onSwitchToLogin} />);
    await userEvent.type(screen.getByLabelText('Full Name'), 'Test User');
    await userEvent.type(screen.getByLabelText('Email Address'), 'test@example.com');
    await userEvent.type(screen.getByLabelText('Password'), 'password123');
    await userEvent.type(screen.getByLabelText('Confirm Password'), 'password123');
    const form = container.querySelector('form');
    fireEvent.submit(form!);
    expect(await screen.findByText('Email already exists')).toBeInTheDocument();
    expect(onLogin).not.toHaveBeenCalled();
  });

  it('calls onSwitchToLogin when log in link is clicked', async () => {
    render(<SignUp onLogin={onLogin} onSwitchToLogin={onSwitchToLogin} />);
    await userEvent.click(screen.getByText('Log in'));
    expect(onSwitchToLogin).toHaveBeenCalled();
  });
});